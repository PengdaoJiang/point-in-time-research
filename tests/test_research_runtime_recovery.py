from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qplatform.research_checkpoints import CheckpointStore
from qplatform.research_readiness import file_digest, read_json
from qplatform.research_runtime import _atomic, _event, _root, request_stop, run_manifest, runtime_status, validate_manifest

FIXTURE = ROOT / "tools" / "fixtures" / "research_runtime_worker.py"


class AtomicPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "progress.json"
        _atomic(self.path, {"completed_units": 1})

    def tearDown(self):
        self.temp.cleanup()

    @unittest.skipUnless(os.name == "nt", "Windows reader denies delete sharing")
    def test_real_reader_conflict_retries_publication_without_recomputing(self):
        with self.path.open("rb") as reader, \
                patch("qplatform.research_runtime.time.sleep", side_effect=lambda delay: reader.close()) as sleep:
            _atomic(self.path, {"completed_units": 2})
        sleep.assert_called_once_with(.01)
        self.assertEqual(read_json(self.path), {"completed_units": 2})
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows reader denies delete sharing")
    def test_persistent_reader_conflict_is_bounded_and_preserves_old_snapshot(self):
        with self.path.open("rb"), patch("qplatform.research_runtime.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                _atomic(self.path, {"completed_units": 2})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [.01, .02, .04, .08, .1])
        self.assertEqual(read_json(self.path), {"completed_units": 1})
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_unrelated_replace_error_does_not_retry_or_erase_old_snapshot(self):
        with patch("qplatform.research_runtime.os.replace", side_effect=OSError("disk failure")) as replace, \
                patch("qplatform.research_runtime.time.sleep") as sleep:
            with self.assertRaisesRegex(OSError, "disk failure"):
                _atomic(self.path, {"completed_units": 2})
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(read_json(self.path), {"completed_units": 1})
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_invalid_json_never_replaces_valid_snapshot(self):
        with patch("qplatform.research_runtime.os.replace") as replace:
            with self.assertRaises(ValueError):
                _atomic(self.path, {"completed_units": float("nan")})
        replace.assert_not_called()
        self.assertEqual(read_json(self.path), {"completed_units": 1})
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.path = self.workspace / "manifest.json"
        self.manifest = {
            "schema": "research_runtime_v1", "run_id": "recovery_test",
            "bindings": {"worker": {"path": str(FIXTURE), "sha256": file_digest(FIXTURE), "role": "science"}},
            "stages": [self.stage("units", "units_crash_once")],
        }
        self.root = _root(self.workspace, "recovery_test")

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, name, mode):
        return {"id": name, "phase": "validation", "uses": ["worker"],
                "command": ["{python}", str(FIXTURE), mode], "outputs": ["result.json"],
                "checkpoint_units": mode.startswith("units"),
                "resources": {"max_memory_bytes": 512 * 1024**2, "available_floor_bytes": 1,
                              "wait_timeout_s": .5, "wall_timeout_s": 10, "stop_grace_s": 2}}

    def run_plan(self):
        _atomic(self.path, self.manifest)
        result = run_manifest(self.workspace, self.path, poll_s=.03, expected_run_id="recovery_test")
        if result.get("attempt"):
            stderr = Path(result["attempt"]) / "stderr.log"
            if stderr.exists():
                result["test_child_stderr"] = stderr.read_text(encoding="utf-8", errors="replace")
        return result

    def output(self, status, stage="units"):
        return read_json(Path(status["stages"][stage]["receipt"]).parent / "artifacts" / "result.json")

    def computed_count(self):
        return len((self.workspace / "computed_units.jsonl").read_text().splitlines())

    def test_crash_resume_reuses_unit_without_new_code_and_keeps_negative_accounts(self):
        self.manifest["stages"][0]["progress_file"] = "progress.json"
        failed = self.run_plan()
        self.assertIn("child_exit:17", failed["blocker"])
        self.assertEqual(self.computed_count(), 1)
        completed = self.run_plan()
        self.assertEqual(completed["status"], "execution_complete", completed)
        self.assertEqual(self.computed_count(), 3)
        self.assertEqual(self.output(completed)["reused_units"], 1)
        self.assertEqual([r["equity"] for r in self.output(completed)["accounts"]], [149900, 149800, 149700])
        self.assertEqual(completed["completed_units"], 3)
        self.assertEqual(completed["manifest_sha256"], file_digest(self.path))
        self.assertFalse(completed["research_complete"])
        self.assertFalse(completed["promotion_authority"])

    def test_resource_only_amendment_reuses_sealed_units_but_revalidates_stage(self):
        self.run_plan()
        self.manifest["stages"][0]["resources"]["available_floor_bytes"] = 2
        result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)
        self.assertEqual(self.output(result)["reused_units"], 1)
        self.assertEqual(self.computed_count(), 3)

    def test_live_progress_reader_does_not_crash_checkpoint_producer(self):
        self.manifest["stages"][0]["progress_file"] = "progress.json"

        def slow_read(path):
            if Path(path).name != "progress.json":
                return read_json(path)
            with Path(path).open("r", encoding="utf-8-sig") as reader:
                time.sleep(.05)
                return json.load(reader)

        with patch("qplatform.research_runtime.read_json", side_effect=slow_read):
            self.assertIn("child_exit:17", self.run_plan()["blocker"])
            result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)
        self.assertEqual(result["completed_units"], 3)
        self.assertEqual(self.computed_count(), 3)
        self.assertEqual(self.output(result)["reused_units"], 1)

    def test_controller_crash_reuses_sealed_unit_with_same_command(self):
        self.manifest["stages"][0]["command"][-1] = "units_controller_crash_once"
        _atomic(self.path, self.manifest)
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "qpt.py"), "research-runtime", "run",
            "--run-id", "recovery_test", "--workspace", str(self.workspace), "--manifest", str(self.path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 10
            while not (self.workspace / "failure_injected").exists() and time.monotonic() < deadline:
                time.sleep(.03)
            self.assertTrue((self.workspace / "failure_injected").exists())
            process.kill()
            process.wait(timeout=10)
            self.assertEqual(runtime_status(self.workspace, "recovery_test")["status"], "interrupted")
            result = self.run_plan()
            self.assertEqual(result["status"], "execution_complete", result)
            self.assertEqual(self.output(result)["reused_units"], 1)
            self.assertEqual(self.computed_count(), 3)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()

    def test_recovery_accounts_match_uninterrupted_same_policy(self):
        self.run_plan()
        recovered = self.output(self.run_plan())["accounts"]
        (self.workspace / "failure_injected").unlink()
        self.manifest["run_id"] = "control"
        (self.workspace / "failure_injected").touch()
        _atomic(self.path, self.manifest)
        result = run_manifest(self.workspace, self.path, poll_s=.03, expected_run_id="control")
        self.assertEqual(self.output(result)["accounts"], recovered)
        self.assertEqual(self.output(result)["reused_units"], 0)

    def test_changed_source_cannot_reuse_prior_unit(self):
        source = self.workspace / "source.json"
        source.write_text('{"v":1}')
        self.manifest["bindings"]["source"] = {"path": str(source), "sha256": file_digest(source), "role": "source"}
        self.manifest["stages"][0]["uses"].append("source")
        self.run_plan()
        source.write_text('{"v":2}')
        self.manifest["bindings"]["source"]["sha256"] = file_digest(source)
        result = self.run_plan()
        self.assertEqual(self.output(result)["reused_units"], 0)
        self.assertEqual(self.computed_count(), 4)

    def test_changed_command_cannot_reuse_prior_unit(self):
        self.run_plan()
        self.manifest["stages"][0]["command"][-1] = "units"
        result = self.run_plan()
        self.assertEqual(self.output(result)["reused_units"], 0)
        self.assertEqual(self.computed_count(), 4)

    def test_checkpoint_corruption_fails_closed_without_silent_recompute(self):
        self.run_plan()
        path = next((self.root / "checkpoints").glob("*/*/*/artifacts/account.json"))
        path.write_text('{"equity":999999999}')
        result = self.run_plan()
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(self.computed_count(), 1)
        stderr = (Path(result["attempt"]) / "stderr.log").read_text()
        self.assertIn("checkpoint_artifact_mismatch", stderr)

    def test_incomplete_pending_unit_never_reused(self):
        store = CheckpointStore(self.workspace / "units", "a" * 64)
        pending = store.root / ".pending-interrupted"
        pending.mkdir(parents=True)
        (pending / "partial.json").write_text('{"partial":true}')
        self.assertIsNone(store.load("unit"))

    def test_immutable_unit_and_path_escape(self):
        store = CheckpointStore(self.workspace / "units", "a" * 64)
        source = self.workspace / "data"
        source.mkdir()
        (source / "unit.json").write_text('{"equity":149999}')
        store.seal("unit", source, ["unit.json"])
        (source / "unit.json").write_text('{"equity":150001}')
        with self.assertRaisesRegex(ValueError, "immutable_conflict"):
            store.seal("unit", source, ["unit.json"])
        with self.assertRaises(ValueError):
            store.seal("escape", source, ["../outside.json"])

    def test_expected_run_id_prevents_old_run_dispatch(self):
        _atomic(self.path, self.manifest)
        with self.assertRaisesRegex(ValueError, "manifest_run_identity_mismatch"):
            run_manifest(self.workspace, self.path, expected_run_id="another_run")
        self.assertFalse(self.root.exists())

    def test_cli_rejects_wrong_run_before_launch(self):
        _atomic(self.path, self.manifest)
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "qpt.py"), "research-runtime", "run",
            "--run-id", "wrong_run", "--workspace", str(self.workspace), "--manifest", str(self.path)],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("manifest_run_identity_mismatch", result.stdout)
        self.assertFalse(self.root.exists())

    def test_current_identity_mismatch_cannot_report_completion(self):
        _atomic(self.root / "current.json", {"run_id": "old_run", "status": "execution_complete", "research_complete": True})
        status = runtime_status(self.workspace, "recovery_test", compact=True)
        self.assertEqual(status["status"], "invalid_state")
        self.assertFalse(status["execution_complete"])
        with self.assertRaisesRegex(ValueError, "current_run_identity_mismatch"):
            self.run_plan()

    def test_compact_status_excludes_historical_fields(self):
        _atomic(self.root / "current.json", {"run_id": "recovery_test", "status": "failed",
            "research_complete": True, "error": "historical", "current_execution_v6": {"status": "running"}})
        status = runtime_status(self.workspace, "recovery_test", compact=True)
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["research_complete"])
        self.assertNotIn("error", status)
        self.assertNotIn("current_execution_v6", status)

    def test_failed_branch_does_not_block_independent_work_or_turn_into_cash(self):
        first = self.stage("broken", "fail")
        descendant = self.stage("dependent", "scale")
        descendant["depends_on"] = ["broken"]
        self.manifest.update(failure_policy="continue_independent", stages=[first, descendant, self.stage("healthy", "scale")])
        result = self.run_plan()
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(result["stage_counts"]["failed"], 1)
        self.assertEqual(result["stage_counts"]["blocked"], 1)
        self.assertEqual(result["stage_counts"]["succeeded"], 1)
        self.assertFalse((self.root / "stages" / "dependent").exists())
        self.assertFalse(result["execution_complete"])
        self.assertEqual(self.output(result, "healthy"), {"computed": True})
        compact = runtime_status(self.workspace, "recovery_test", compact=True)
        self.assertEqual([row["stage"] for row in compact["blocking_stages"]], ["broken", "dependent"])

    def test_user_stop_overrides_independent_continuation(self):
        self.manifest.update(failure_policy="continue_independent",
                             stages=[self.stage("broken", "fail"), self.stage("healthy", "scale")])
        def deliver_then_stop(root, payload):
            event_id = _event(root, payload)
            if payload["status"] == "failed":
                request_stop(self.workspace, "recovery_test")
            return event_id
        with patch("qplatform.research_runtime._event", side_effect=deliver_then_stop):
            result = self.run_plan()
        self.assertEqual(result["status"], "stopped", result)
        self.assertFalse((self.root / "stages" / "healthy").exists())
        self.assertTrue((self.root / "stop.json").exists())

    def test_source_unavailable_is_bound_and_kept_in_scope_before_launch(self):
        evidence = self.workspace / "capability.json"
        evidence.write_text('{"exact_volume":null,"reason":"source rounded"}')
        self.manifest["bindings"]["capability"] = {"path": str(evidence), "sha256": file_digest(evidence), "role": "contract"}
        held = self.stage("exact_volume", "fail")
        held["uses"].append("capability")
        held["unavailable"] = {"reason": "exact volume unavailable", "evidence_binding": "capability"}
        self.manifest["stages"] = [held, self.stage("healthy", "scale")]
        result = self.run_plan()
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(result["stage_counts"]["unavailable"], 1)
        self.assertFalse((self.root / "stages" / "exact_volume").exists())
        self.assertEqual(result["stage_outcomes"]["exact_volume"]["reason"], "exact volume unavailable")
        evidence.write_text("changed")
        result = self.run_plan()
        self.assertIn("input_binding_mismatch:capability", result["blocker"])

    def memory_stage(self, mode="progress"):
        stage = self.stage("memory", mode)
        stage["resources"].update(available_floor_bytes=6 * 1024**3, wall_timeout_s=5)
        stage["memory_pressure"] = {"grace_s": .12, "critical_available_bytes": 1024**3}
        self.manifest["stages"] = [stage]
        return stage

    def memory_source(self, low, *, recover=False):
        started = 0
        def sample():
            nonlocal started
            if not list(self.root.glob("stages/*/attempts/*/process.json")):
                return SimpleNamespace(available=8 * 1024**3)
            started += 1
            return SimpleNamespace(available=8 * 1024**3 if recover and started > 2 else low)
        return sample

    def test_transient_host_pressure_does_not_terminate_valid_work(self):
        self.memory_stage()
        with patch("qplatform.research_runtime.psutil.virtual_memory", side_effect=self.memory_source(6 * 1024**3 - 24 * 1024**2, recover=True)):
            result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)

    def test_sustained_host_pressure_requests_cooperative_checkpoint(self):
        self.memory_stage("cooperative")
        with patch("qplatform.research_runtime.psutil.virtual_memory", side_effect=self.memory_source(5 * 1024**3)):
            result = self.run_plan()
        self.assertEqual(result["blocker"], "host_memory_pressure", result)
        self.assertTrue((Path(result["attempt"]) / "artifacts" / "checkpoint.json").exists())
        self.assertTrue(read_json(Path(result["attempt"]) / "process_result.json")["owned_processes_drained"])

    def test_critical_host_pressure_still_stops_immediately(self):
        self.memory_stage("hang")
        with patch("qplatform.research_runtime.psutil.virtual_memory", side_effect=self.memory_source(512 * 1024**2)):
            result = self.run_plan()
        self.assertEqual(result["blocker"], "host_memory_critical", result)
        self.assertFalse((Path(result["attempt"]) / "receipt.json").exists())

    def test_invalid_memory_policy_and_unbound_unavailable_rejected(self):
        self.memory_stage()["memory_pressure"]["critical_available_bytes"] = 8 * 1024**3
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)
        self.manifest["stages"] = [self.stage("bad", "scale")]
        self.manifest["stages"][0]["unavailable"] = {"reason": "missing", "evidence_binding": "unbound"}
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
