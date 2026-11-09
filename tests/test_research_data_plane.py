from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qplatform.research_readiness import file_digest, read_json
from qplatform.research_runtime import run_manifest, request_stop, _inputs
from qplatform.research_store import ResearchTable, freeze_table
from qplatform.research_batch import run_batch, validate_plan, verify_results
from qplatform.research_checkpoints import CheckpointStore

SEMANTICS = {"universe": "test_only", "price_basis": "raw", "frequency": "daily",
             "availability": "declared fixture dates, not PIT certification", "role": "test"}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.parquet"
        self.frame = pd.DataFrame({
            "symbol": ["000001", "300001", "000001", "300001"],
            "date": ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02"],
            "x": [1., 2., float("nan"), 4.], "unused": [7, 8, 9, 10]})
        self.frame.to_parquet(self.source, index=False)
        self.store = self.root / "store"

    def freeze(self, **kwargs):
        return freeze_table(self.store, [self.source], SEMANTICS, **kwargs)

    def test_exact_copy_dedup_and_source_change_isolation(self):
        first = self.freeze()
        second = self.freeze()
        self.assertEqual(first["snapshot"], second["snapshot"])
        self.assertEqual(second["added_bytes"], 0)
        self.assertTrue(second["freeze_cache_hit"])
        original = ResearchTable(Path(first["snapshot"]))
        self.frame.assign(x=99.).to_parquet(self.source, index=False)
        pd.testing.assert_frame_equal(original.read(), self.frame)
        self.assertNotEqual(self.freeze()["snapshot_id"], first["snapshot_id"])

    def test_partition_projection_filter_empty_and_batch_parity(self):
        result = self.freeze(partition_by_date=True)
        table = ResearchTable(Path(result["snapshot"]))
        query = {"columns": ["symbol", "date", "x"], "dates": ["2026-07-02"], "symbols": ["300001"]}
        pd.testing.assert_frame_equal(table.read(**query).reset_index(drop=True), self.frame.iloc[[3]][query["columns"]].reset_index(drop=True))
        self.assertEqual(table.last_scan["files_selected"], 1)
        self.assertEqual(table.last_scan["files_total"], 2)
        batches = list(table.scanner(**query, batch_size=1).to_batches())
        self.assertEqual(sum(b.num_rows for b in batches), 1)
        self.assertEqual(len(table.read(dates=[])), 0)
        with self.assertRaises(ValueError):
            table.read(columns=["missing"])
        with self.assertRaises(ValueError):
            table.read(dates=["2026-7-2"])
        with self.assertRaises(ValueError):
            table.read(symbols=[300001])

    def test_append_reuses_unchanged_date_objects(self):
        first = self.freeze(partition_by_date=True)
        extra = self.frame.iloc[:1].assign(date="2026-07-03")
        pd.concat([self.frame, extra]).to_parquet(self.source, index=False)
        second = self.freeze(partition_by_date=True)
        self.assertEqual(second["objects"], 3)
        self.assertGreater(second["reused_bytes"], 0)
        old = read_json(Path(first["snapshot"]))
        new = read_json(Path(second["snapshot"]))
        self.assertEqual([o["sha256"] for o in old["objects"]], [o["sha256"] for o in new["objects"][:2]])

    def test_corruption_rejected_in_reader_and_runtime_binding(self):
        result = self.freeze()
        path = Path(result["snapshot"])
        reader = ResearchTable(path)
        reader.verify_all()
        obj = next((self.store / "objects").rglob("*.parquet"))
        with obj.open("ab") as stream:
            stream.write(b"bad")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            reader.read()
        manifest = {"bindings": {"s": {"path": str(path), "sha256": file_digest(path), "kind": "research_snapshot"}}}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            _inputs(self.root, manifest, {"uses": ["s"]})

    def test_semantic_identity_and_validation(self):
        first = self.freeze()
        second = freeze_table(self.store, [self.source], {**SEMANTICS, "price_basis": "front_adjusted"})
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertEqual(second["added_bytes"], 0)
        with self.assertRaises(ValueError):
            freeze_table(self.store, [self.source], {})
        with self.assertRaisesRegex(ValueError, "budget"):
            self.freeze(partition_by_date=True, max_materialize_bytes=1)
        with self.assertRaises(ValueError):
            freeze_table(self.store, [self.source, self.source], SEMANTICS)

    def test_empty_and_null_dates(self):
        # An empty object column is inferred as Arrow null on some pandas
        # versions. Declare the string schema that the partition contract needs.
        self.frame.iloc[:0].astype({"date": "string"}).to_parquet(self.source, index=False)
        reader = ResearchTable(Path(self.freeze(partition_by_date=True)["snapshot"]))
        self.assertTrue(reader.read().empty)
        self.frame.assign(date=None).to_parquet(self.source, index=False)
        with self.assertRaises(ValueError):
            self.freeze(partition_by_date=True)

    def test_protected_store_and_schema_drift_rejected(self):
        with self.assertRaises(ValueError):
            freeze_table(self.root / "cross_section" / "store", [self.source], SEMANTICS)
        other = self.root / "other.parquet"
        self.frame.assign(extra=1).to_parquet(other, index=False)
        with self.assertRaisesRegex(ValueError, "schemas"):
            freeze_table(self.store, [self.source, other], SEMANTICS)

    def test_input_change_during_freeze_never_publishes_manifest(self):
        from qplatform import research_store
        original = research_store.shutil.copyfile
        def changing(src, dest):
            original(src, dest)
            self.frame.assign(x=99.).to_parquet(src, index=False)
        with patch.object(research_store.shutil, "copyfile", side_effect=changing):
            with self.assertRaises(ValueError):
                self.freeze()
        self.assertFalse(list((self.store / "snapshots").glob("*.json")))



class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = self.root / "spec.json"
        self.spec.write_text('{"purpose":"infrastructure test only"}')
        self.plan_path = self.root / "batch.json"
        self.fixture = ROOT / "tools" / "fixtures" / "research_batch_worker.py"
        self.plan = {"schema": "research_parallel_batch_v1", "question": "serial/parallel parity",
                     "research_spec_binding": "spec", "worker_memory_bytes": 64 * 1024**2,
                     "worker_threads": 1, "jobs": [
                         {"id": f"job{i}", "role": "diagnostic", "entrypoint_binding": "worker",
                          "args": ["--value", str(i), "--delay", ".2"], "outputs": ["result.json"]} for i in range(4)]}

    def manifest(self, run_id):
        self.plan_path.write_text(json.dumps(self.plan))
        bindings = {key: {"path": str(path), "sha256": file_digest(path), "role": role} for key, path, role in (
            ("plan", self.plan_path, "science"), ("worker", self.fixture, "entrypoint"), ("spec", self.spec, "contract"))}
        manifest = {"schema": "research_runtime_v1", "run_id": run_id, "bindings": bindings,
                    "stages": [{"id": "batch", "phase": "validation", "uses": list(bindings),
                        "command": ["{python}", "-m", "qplatform.research_batch", "--plan", str(self.plan_path)],
                        "outputs": ["batch_results.json"], "parallel_batch": True, "checkpoint_units": True,
                        "progress_file": "progress.json",
                        "resources": {"max_memory_bytes": 512 * 1024**2, "available_floor_bytes": 16 * 1024**2,
                                      "wait_timeout_s": 5, "wall_timeout_s": 30, "stop_grace_s": 2}}]}
        path = self.root / f"{run_id}.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def results(self, run_id):
        runtime = self.root / "workspace_out" / "runs" / run_id / "operations" / "research_runtime"
        files = sorted(runtime.glob("stages/batch/attempts/*/artifacts/batch_results.json"))
        return runtime, read_json(files[-1])

    def test_parallel_and_serial_parity_and_reuse(self):
        for run_id, slots in (("serial", "1"), ("parallel", "3")):
            path, _ = self.manifest(run_id)
            with patch.dict(os.environ, {"QPT_RESEARCH_WORKERS": slots}):
                result = run_manifest(self.root, path, poll_s=.05)
            self.assertEqual(result["status"], "execution_complete", result)
        serial_root, serial = self.results("serial")
        runtime, parallel = self.results("parallel")
        self.assertEqual(serial["peak_workers"], 1)
        self.assertGreater(parallel["peak_workers"], 1)
        self.assertEqual(sorted(serial["jobs"]), sorted(parallel["jobs"]))
        for directory, result in ((serial_root, serial), (runtime, parallel)):
            store = CheckpointStore(directory / "checkpoints/batch", result["work_binding_id"])
            for i in range(4):
                loaded = store.load(f"job{i}")
                self.assertEqual(read_json(loaded["directory"] / "result.json"),
                                 {"value": i**2, "threads": "1"})
        result = run_manifest(self.root, path, poll_s=.05)
        self.assertEqual(result["status"], "execution_complete")
        self.assertTrue(result["stage_outcomes"]["batch"]["reused"])
        # No silent reuse of a manifest whose transitive unit output was corrupted.
        artifact = next((runtime / "checkpoints").rglob("result.json"))
        artifact.write_text('{"value":999}')
        result = run_manifest(self.root, path, poll_s=.05)
        self.assertNotEqual(result["status"], "execution_complete")

    def test_failure_retains_siblings_and_resume_reuses_units(self):
        marker = self.root / "fixed"
        self.plan["jobs"][1]["args"] += ["--fail-unless", str(marker)]
        path, _ = self.manifest("resume")
        first = run_manifest(self.root, path, poll_s=.05)
        self.assertNotEqual(first["status"], "execution_complete")
        _, result = self.results("resume")
        self.assertEqual(len(result["jobs"]), 3)
        marker.touch()
        final = run_manifest(self.root, path, poll_s=.05, resume=True)
        self.assertEqual(final["status"], "execution_complete", final)
        _, result = self.results("resume")
        self.assertEqual(sum(x["reused"] for x in result["jobs"].values()), 3)

    def test_direct_launch_and_unbound_code_rejected(self):
        with patch.dict(os.environ, {"QPT_PARALLEL_BATCH": "0"}):
            with self.assertRaises(ValueError):
                run_batch(self.plan_path)
        _, manifest = self.manifest("invalid")
        bad = copy.deepcopy(self.plan)
        bad["jobs"][0]["entrypoint_binding"] = "not_bound"
        with self.assertRaises(ValueError):
            validate_plan(bad, manifest["bindings"])

    def test_opt_in_bundle_and_owned_scratch_retirement(self):
        self.plan.update(bundle_results=True, discard_sealed_work_outputs=True)
        path, _ = self.manifest('packed')
        outcome = run_manifest(self.root, path, poll_s=.05)
        self.assertEqual(outcome['status'], 'execution_complete', outcome)
        runtime, result = self.results('packed')
        self.assertEqual(result['result_bundle']['files'], 3)
        scratch = list(runtime.glob('stages/batch/attempts/*/artifacts/work/*/result.json'))
        self.assertEqual(scratch, [])
        self.assertEqual(len(list((runtime / 'checkpoints').rglob('result.json'))), 4)
        repeated = run_manifest(self.root, path, poll_s=.05)
        self.assertTrue(repeated['stage_outcomes']['batch']['reused'])
        part = next(runtime.glob('stages/batch/attempts/*/artifacts/result_bundle/part-*.zip'))
        with part.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'part_mismatch'):
            result_path = next(runtime.glob('stages/batch/attempts/*/artifacts/batch_results.json'))
            verify_results(result_path, runtime / 'checkpoints/batch', result['work_binding_id'])

    def test_safe_stop_drains_active_and_resume_reuses(self):
        for job in self.plan["jobs"]:
            job["args"][-1] = ".5"
        path, _ = self.manifest("stop")
        runtime = self.root / "workspace_out/runs/stop/operations/research_runtime"
        outcome = []
        with patch.dict(os.environ, {"QPT_RESEARCH_WORKERS": "1"}):
            thread = threading.Thread(target=lambda: outcome.append(run_manifest(self.root, path, poll_s=.05)))
            thread.start()
            deadline = time.monotonic() + 15
            try:
                while time.monotonic() < deadline:
                    if list((runtime / "checkpoints").rglob("receipt.json")):
                        break
                    time.sleep(.03)
                else:
                    self.fail("no checkpoint before stop deadline")
                request_stop(self.root, "stop")
            finally:
                thread.join(timeout=20)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcome[0]["status"], "stopped")
            retained = len(list((runtime / "checkpoints").rglob("receipt.json")))
            self.assertGreater(retained, 0)
            self.assertLess(retained, 4)
            final = run_manifest(self.root, path, poll_s=.05, resume=True)
        self.assertEqual(final["status"], "execution_complete", final)
        _, result = self.results("stop")
        self.assertEqual(sum(x["reused"] for x in result["jobs"].values()), retained)


if __name__ == "__main__":
    unittest.main()
