from __future__ import annotations

import copy
import errno
import json
import os
import subprocess
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

import psutil
from qplatform.research_readiness import audit_label_support, audit_readiness, compare_scenario_maturity, file_digest, read_json
from qplatform.research_runtime import (
    _atomic, _read_progress, _root, acknowledge_event, request_stop, run_manifest, runtime_status, validate_manifest,
)

FIXTURE = ROOT / "tools" / "fixtures" / "research_runtime_worker.py"
CONTRACT = {"scenarios": {
    "L2": {"mode": "fit", "required_training_roles": ["action"], "maturity_by_window": {"5": 4, "10": 9, "20": 19}},
    "L5": {"mode": "fit", "required_training_roles": ["action"], "maturity_by_window": {"5": 4, "10": 9, "20": 16}},
}}


def access_conflict(code=5):
    error = PermissionError(errno.EACCES, "injected Windows access conflict")
    error.winerror = code
    return error


class ProgressReadTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows file sharing")
    def test_real_windows_exclusive_handle_release_allows_read(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            path.write_text('{"completed_units":3989}')
            handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value, ctypes.get_last_error())
            release = threading.Timer(.06, lambda: kernel.CloseHandle(handle))
            release.start()
            try:
                self.assertEqual(_read_progress(path), {"completed_units": 3989})
            finally:
                release.join()

    @unittest.skipUnless(os.name == "nt", "Windows errno-only PermissionError")
    def test_windows_errno_only_access_denial_is_retried(self):
        error = PermissionError(errno.EACCES, "injected errno-only denial")
        expected = {"completed_units": 1}
        with patch("qplatform.research_runtime.read_json", side_effect=[error, expected]), \
                patch("qplatform.research_runtime.time.sleep") as sleep:
            self.assertEqual(_read_progress(Path("progress.json")), expected)
        sleep.assert_called_once_with(.01)

    def test_transient_windows_access_errors_retry_exact_read(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                expected = {"completed_units": 3989}
                with patch("qplatform.research_runtime.read_json", side_effect=[access_conflict(code), expected]) as reader, \
                        patch("qplatform.research_runtime.time.sleep") as sleep:
                    self.assertEqual(_read_progress(Path("progress.json")), expected)
                self.assertEqual(reader.call_count, 2)
                sleep.assert_called_once_with(.01)

    def test_persistent_access_error_is_bounded_and_not_suppressed(self):
        error = access_conflict()
        with patch("qplatform.research_runtime.read_json", side_effect=error) as reader, \
                patch("qplatform.research_runtime.time.sleep") as sleep:
            with self.assertRaises(PermissionError) as caught:
                _read_progress(Path("progress.json"))
        self.assertIs(caught.exception, error)
        self.assertEqual(reader.call_count, 6)
        self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), .25)

    def test_unrelated_io_and_invalid_json_are_not_retried(self):
        for error in (OSError(errno.EIO, "injected disk error"), json.JSONDecodeError("bad JSON", "{", 1)):
            with self.subTest(error=type(error).__name__):
                with patch("qplatform.research_runtime.read_json", side_effect=error) as reader, \
                        patch("qplatform.research_runtime.time.sleep") as sleep:
                    with self.assertRaises(type(error)):
                        _read_progress(Path("progress.json"))
                reader.assert_called_once()
                sleep.assert_not_called()


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.manifest_path = self.workspace / "manifest.json"
        self.manifest = {
            "schema": "research_runtime_v1", "run_id": "fault_test",
            "bindings": {"fixture": {"path": str(FIXTURE), "sha256": file_digest(FIXTURE), "role": "science"}},
            "stages": [self.stage("pilot", "pilot"), self.stage("scale", "scale")],
        }
        self.root = _root(self.workspace, "fault_test")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def stage(self, name: str, mode: str) -> dict:
        stage = {
            "id": name, "phase": name, "uses": ["fixture"],
            "command": ["{python}", str(FIXTURE), mode], "outputs": ["evidence.json" if name == "pilot" else "result.json"],
            "resources": {"max_memory_bytes": 512 * 1024**2, "available_floor_bytes": 1,
                          "wait_timeout_s": .15, "wall_timeout_s": 10, "stop_grace_s": .3},
        }
        if name == "pilot":
            stage["readiness_contract"] = copy.deepcopy(CONTRACT)
        else:
            stage.update(pilot_stage="pilot", depends_on=["pilot"])
        return stage

    def run_plan(self) -> dict:
        _atomic(self.manifest_path, self.manifest)
        return run_manifest(self.workspace, self.manifest_path, poll_s=.03)

    def progress_stage(self) -> None:
        stage = self.stage("scale", "progress")
        stage.update(id="progress", phase="validation", progress_file="progress.json")
        stage.pop("pilot_stage")
        stage.pop("depends_on")
        self.manifest["stages"] = [stage]

    def pilot_attempt(self, result: dict) -> Path:
        return Path(result["stages"]["pilot"]["receipt"]).parent

    def child_controller(self) -> subprocess.Popen:
        _atomic(self.manifest_path, self.manifest)
        return subprocess.Popen([sys.executable, str(ROOT / "scripts" / "qpt.py"), "research-runtime", "run",
                                 "--run-id", "fault_test",
                                 "--workspace", str(self.workspace), "--manifest", str(self.manifest_path)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def wait_for(self, predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.03)
        self.fail("timed out waiting for test child")

    def test_actual_pilot_scale_negative_returns_and_reuse(self):
        result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)
        self.assertFalse(result["research_complete"])
        receipt = json.loads(Path(result["stages"]["pilot"]["receipt"]).read_text())
        self.assertEqual(receipt["readiness"]["scenarios"]["L5"]["actual_training_rows"], 2)
        second = self.run_plan()
        self.assertTrue(all(row["reused"] for row in second["stages"].values()))
        self.assertEqual(len(second["pending_events"]), 1)

    def test_transient_progress_read_denial_does_not_kill_real_child(self):
        self.progress_stage()
        failures = 0

        def read_with_conflict(path):
            nonlocal failures
            if Path(path).name == "progress.json" and failures < 2:
                failures += 1
                raise access_conflict()
            return read_json(path)

        with patch("qplatform.research_runtime.read_json", side_effect=read_with_conflict):
            result = self.run_plan()
        self.assertEqual(failures, 2)
        self.assertEqual(result["status"], "execution_complete", result)
        receipt = read_json(Path(result["stages"]["progress"]["receipt"]))
        self.assertEqual(receipt["process"]["completed_units"], 1)
        self.assertTrue(receipt["process"]["owned_processes_drained"])
        self.assertIsNone(receipt["process"]["progress_error"])

    def test_permanent_progress_denial_records_drained_process_failure(self):
        self.progress_stage()

        def denied_progress(path):
            if Path(path).name == "progress.json":
                raise access_conflict()
            return read_json(path)

        with patch("qplatform.research_runtime.read_json", side_effect=denied_progress):
            result = self.run_plan()
        self.assertEqual(result["blocker"], "progress_read_failed:PermissionError", result)
        attempt = Path(result["attempt"])
        child = read_json(attempt / "process_result.json")
        self.assertTrue(child["owned_processes_drained"])
        self.assertIsNone(child["completed_units"])
        self.assertEqual(child["progress_error"]["winerror"], 5)
        self.assertTrue((attempt / "failure.json").exists())
        self.assertFalse((attempt / "receipt.json").exists())
        self.assertFalse((self.root / "stages" / "progress" / "latest_success.json").exists())

    def test_invalid_progress_still_fails_with_cleanup_evidence(self):
        self.progress_stage()

        def invalid_progress(path):
            if Path(path).name == "progress.json":
                raise json.JSONDecodeError("bad JSON", "{", 1)
            return read_json(path)

        with patch("qplatform.research_runtime.read_json", side_effect=invalid_progress):
            result = self.run_plan()
        self.assertEqual(result["blocker"], "progress_read_failed:JSONDecodeError", result)
        child = read_json(Path(result["attempt"]) / "process_result.json")
        self.assertTrue(child["owned_processes_drained"])
        self.assertFalse((Path(result["attempt"]) / "receipt.json").exists())

    def test_regressing_progress_preserves_last_good_count(self):
        self.progress_stage()
        counts = iter((2, 1))

        def regressing_progress(path):
            if Path(path).name == "progress.json":
                return {"completed_units": next(counts)}
            return read_json(path)

        with patch("qplatform.research_runtime.read_json", side_effect=regressing_progress):
            result = self.run_plan()
        self.assertEqual(result["blocker"], "nonmonotonic_progress", result)
        child = read_json(Path(result["attempt"]) / "process_result.json")
        self.assertTrue(child["owned_processes_drained"])
        self.assertEqual(child["completed_units"], 2)

    def test_scope_local_maturity_not_global_max(self):
        expected = {k: v["maturity_by_window"] for k, v in CONTRACT["scenarios"].items()}
        self.assertEqual(compare_scenario_maturity(expected, copy.deepcopy(expected)), [])
        wrong = {k: {"5": 4, "10": 9, "20": 19} for k in expected}
        self.assertIn("L5:maturity_reference_mismatch", compare_scenario_maturity(expected, wrong))

    def test_target_workspace_package_cannot_shadow_worktree_bootstrap(self):
        package = self.workspace / "qplatform"
        package.mkdir()
        (package / "__init__.py").write_text("")
        result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)

    def test_zero_training_blocks_scale_not_negative_strategy(self):
        self.manifest["stages"][0]["command"][-1] = "empty_fit"
        result = self.run_plan()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["category"], "learning_or_evidence_not_ready")
        self.assertIn("no_effective_learning", result["blocker"])
        self.assertFalse((self.root / "stages" / "scale").exists())
        audit = json.loads((Path(result["attempt"]) / "readiness_audit.json").read_text())
        self.assertEqual(audit["scenarios"]["L2"]["unresolved"], 2)
        self.assertFalse(audit["strategy_profitability_assessed"])

    def test_future_label_blocks_scale(self):
        self.manifest["stages"][0]["command"][-1] = "future"
        self.assertIn("fit_consumed_unobservable_label", self.run_plan()["blocker"])

    def test_wait_only_fit_is_not_action_learning(self):
        self.manifest["stages"][0]["command"][-1] = "wait_only"
        result = self.run_plan()
        self.assertIn("no_training_support_for_role:action", result["blocker"])
        audit = json.loads((Path(result["attempt"]) / "readiness_audit.json").read_text())
        self.assertEqual(audit["scenarios"]["L2"]["actual_training_rows"], 2)
        self.assertEqual(audit["scenarios"]["L2"]["training_rows_by_role"], {"wait": 2})

    def test_failure_receipt_and_explicit_delivery_ack(self):
        self.manifest["stages"][0]["command"][-1] = "fail"
        result = self.run_plan()
        self.assertIn("child_exit:", result["blocker"])
        self.assertTrue((Path(result["attempt"]) / "failure.json").exists())
        self.assertEqual(len(result["pending_events"]), 1)
        key = Path(result["pending_events"][0]).stem
        self.assertFalse(json.loads(Path(result["pending_events"][0]).read_text())["delivered"])
        acknowledge_event(self.workspace, "fault_test", key, "unit_test")
        self.assertEqual(runtime_status(self.workspace, "fault_test")["pending_events"], [])

    def test_repeated_failures_do_not_exhaust_repair_attempts(self):
        self.manifest["stages"][0]["command"][-1] = "fail"
        attempts = set()
        for _ in range(3):
            result = self.run_plan()
            self.assertIn("child_exit:", result["blocker"])
            attempts.add(result["attempt"])
        self.assertEqual(len(attempts), 3)
        self.manifest["stages"][0]["command"][-1] = "pilot"
        result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)
        self.assertEqual(len(list((self.root / "stages" / "pilot" / "attempts").iterdir())), 4)
        self.assertTrue(all((Path(path) / "failure.json").exists() for path in attempts))

    def test_rule_replay_does_not_require_learning(self):
        self.manifest["stages"][0]["command"][-1] = "rule_replay"
        for spec in self.manifest["stages"][0]["readiness_contract"]["scenarios"].values():
            spec["mode"] = "replay"
            spec.pop("required_training_roles")
        result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)
        receipt = json.loads(Path(result["stages"]["pilot"]["receipt"]).read_text())
        self.assertEqual(receipt["readiness"]["scenarios"]["L2"]["fit_count"], 0)

    def test_wall_timeout_allows_atomic_checkpoint_before_termination(self):
        stage = self.manifest["stages"][0]
        stage["command"][-1] = "cooperative"
        stage["resources"].update(wall_timeout_s=1.5, stop_grace_s=2)
        result = self.run_plan()
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["category"], "engineering_failure")
        self.assertEqual(result["blocker"], "wall_timeout")
        attempt = Path(result["attempt"])
        self.assertTrue((attempt / "artifacts" / "checkpoint.json").exists())
        child = json.loads((attempt / "process_result.json").read_text())
        self.assertEqual(child["returncode"], 0)
        self.assertTrue(child["owned_processes_drained"])
        self.assertEqual(child["stop_reason"], "wall_timeout")
        self.assertFalse((attempt / "receipt.json").exists())
        self.assertFalse((self.root / "stop.json").exists())
        self.assertFalse((self.root / "stages" / "scale").exists())

    def test_timeout_drains_owned_grandchild(self):
        stage = self.manifest["stages"][0]
        stage["command"][-1] = "spawn"
        stage["resources"]["wall_timeout_s"] = 1.5
        result = self.run_plan()
        self.assertEqual(result["blocker"], "wall_timeout")
        attempt = Path(result["attempt"])
        pid = json.loads((attempt / "artifacts" / "grandchild.json").read_text())["pid"]
        self.assertFalse(psutil.pid_exists(pid))
        self.assertTrue(json.loads((attempt / "process_result.json").read_text())["owned_processes_drained"])

    def test_resource_wait_is_bounded_without_launch(self):
        with patch("qplatform.research_runtime.psutil.virtual_memory") as memory:
            memory.return_value.available = 1
            result = self.run_plan()
        self.assertEqual(result["blocker"], "resource_wait_timeout")
        self.assertFalse((Path(result["attempt"]) / "process.json").exists())

    def test_serial_admission_does_not_reserve_finished_stage(self):
        with patch("qplatform.research_runtime.psutil.virtual_memory") as memory:
            memory.return_value.available = 600 * 1024**2
            result = self.run_plan()
        self.assertEqual(result["status"], "execution_complete", result)

    def test_wait_only_label_support_before_any_fit_or_scale(self):
        rows = [{"status": "RESOLVED_MARKET", "value": 0, "training_role": "wait",
                 "available_at": "2026-08-14T15:00:00+08:00"},
                {"status": "UNRESOLVED", "value": None, "training_role": "action", "available_at": ""}]
        result = audit_label_support(rows, cutoff="2026-08-17T09:30:00+08:00", required_roles=["action"])
        self.assertFalse(result["support_available"])
        self.assertEqual(result["observable_labels_by_role"], {"wait": 1})
        self.assertEqual(result["unresolved"], 1)
        self.assertFalse(result["actual_fit_verified"])
        rows[1]["value"] = 0
        result = audit_label_support(rows, cutoff="2026-08-17T09:30:00+08:00", required_roles=["action"])
        self.assertIn("unresolved_label_has_numeric_return", result["blockers"])

    def test_old_concurrent_reservation_is_rejected(self):
        self.manifest["stages"][0]["resources"]["previous_t_reservation"] = 6 * 1024**3
        with self.assertRaisesRegex(ValueError, "only this active stage"):
            validate_manifest(self.manifest)

    def test_input_tamper_prevents_child_launch(self):
        self.manifest["bindings"]["fixture"]["sha256"] = "0" * 64
        result = self.run_plan()
        self.assertIn("input_binding_mismatch", result["blocker"])
        self.assertIsNone(result["attempt"])

    def test_artifact_tamper_invalidates_reuse(self):
        first = self.run_plan()
        (self.pilot_attempt(first) / "artifacts" / "model.json").write_text("corrupted")
        second = self.run_plan()
        self.assertFalse(second["stages"]["pilot"]["reused"])
        self.assertFalse(second["stages"]["scale"]["reused"])

    def test_only_affected_stage_reruns(self):
        self.run_plan()
        self.manifest["stages"][1]["resources"]["wall_timeout_s"] = 11
        second = self.run_plan()
        self.assertTrue(second["stages"]["pilot"]["reused"])
        self.assertFalse(second["stages"]["scale"]["reused"])

    def test_missing_scenario_and_stale_evidence_binding(self):
        result = self.run_plan()
        directory = self.pilot_attempt(result) / "artifacts"
        evidence = json.loads((directory / "evidence.json").read_text())
        evidence["binding_id"] = "stale"
        del evidence["scenarios"]["L5"]
        _atomic(directory / "evidence.json", evidence)
        audit = audit_readiness(directory, CONTRACT, result["stages"]["pilot"]["binding_id"])
        self.assertIn("pilot_binding_mismatch", audit["blockers"])
        self.assertIn("scenario_coverage_mismatch", audit["blockers"])

    def test_no_pilot_and_path_escape_fail_closed(self):
        self.manifest["stages"] = [self.manifest["stages"][1]]
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)
        self.manifest["stages"] = [self.stage("pilot", "pilot")]
        self.manifest["stages"][0]["outputs"] = ["../../bad.json"]
        with self.assertRaisesRegex(ValueError, "escapes"):
            validate_manifest(self.manifest)

    def test_declared_fit_rows_do_not_replace_actual_consumed_ids(self):
        result = self.run_plan()
        directory = self.pilot_attempt(result) / "artifacts"
        evidence = json.loads((directory / "evidence.json").read_text())
        path = directory / evidence["scenarios"]["L2"]["fits"]["path"]
        fit = json.loads(path.read_text())
        fit["fitted_rows"] = 500
        path.write_text(json.dumps(fit))
        evidence["scenarios"]["L2"]["fits"]["sha256"] = file_digest(path)
        _atomic(directory / "evidence.json", evidence)
        audit = audit_readiness(directory, CONTRACT, result["stages"]["pilot"]["binding_id"])
        self.assertIn("L2:invalid_actual_fit_rows", audit["blockers"])

    def test_different_scientific_inputs_require_new_pilot(self):
        self.manifest["bindings"]["other_source"] = {
            "path": str(FIXTURE), "sha256": file_digest(FIXTURE), "role": "source"}
        self.manifest["stages"][1]["uses"].append("other_source")
        with self.assertRaisesRegex(ValueError, "identical scientific inputs"):
            validate_manifest(self.manifest)

    def test_receipt_tamper_never_reuses_success(self):
        first = self.run_plan()
        Path(first["stages"]["scale"]["receipt"]).write_text('{"status":"succeeded"}')
        second = self.run_plan()
        self.assertTrue(second["stages"]["pilot"]["reused"])
        self.assertFalse(second["stages"]["scale"]["reused"])

    def test_stop_does_not_launch_and_requires_explicit_resume(self):
        request_stop(self.workspace, "fault_test")
        result = self.run_plan()
        self.assertEqual(result["status"], "stopped")
        self.assertIsNone(result["attempt"])
        result = run_manifest(self.workspace, self.manifest_path, poll_s=.03, resume=True)
        self.assertEqual(result["status"], "execution_complete")

    def test_cooperative_stop_preserves_complete_checkpoint(self):
        self.manifest["stages"][0]["command"][-1] = "cooperative"
        process = self.child_controller()
        try:
            self.wait_for(lambda: list(self.root.glob("stages/pilot/attempts/*/process.json")))
            request_stop(self.workspace, "fault_test")
            process.wait(timeout=12)
            result = runtime_status(self.workspace, "fault_test")
            self.assertEqual(result["status"], "stopped", result)
            self.assertTrue((Path(result["attempt"]) / "artifacts" / "checkpoint.json").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()

    def test_controller_death_kills_children_and_status_is_not_running(self):
        self.manifest["stages"][0]["command"][-1] = "spawn"
        process = self.child_controller()
        try:
            file = self.wait_for(lambda: list(self.root.glob("stages/pilot/attempts/*/artifacts/grandchild.json")))[0]
            pid = json.loads(file.read_text())["pid"]
            with self.assertRaisesRegex(ValueError, "already owned"):
                self.run_plan()
            process.kill()
            process.wait(timeout=10)
            self.wait_for(lambda: not psutil.pid_exists(pid))
            self.assertEqual(runtime_status(self.workspace, "fault_test")["status"], "interrupted")
            self.manifest["stages"][0]["command"][-1] = "pilot"
            recovered = self.run_plan()
            self.assertEqual(recovered["status"], "execution_complete")
            events = [json.loads(Path(p).read_text()) for p in recovered["pending_events"]]
            self.assertTrue(any(e["status"] == "interrupted" for e in events))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
