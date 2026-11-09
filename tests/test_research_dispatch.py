from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qplatform.control_io import atomic_json
from qplatform.research_readiness import file_digest, read_json
from qplatform.research_runtime import run_manifest, _root
from qplatform.research_observer import build_packet, validate_report, deliver, bounded_cli, StopSignals, acknowledge_delivery
from qplatform.research_dispatch import load_policy, process_event, supervise, dispatch_root, dispatch_status, start_dispatch, stop_dispatch


def report_for(packet):
    return {"event_id": packet["event_id"], "scientific_verdict": "not_assessed", "summary": "Execution evidence only",
            "stage_checks": [{k: s[k] for k in ("id", "status", "evidence_status")} for s in packet["stages"]],
            "questions": []}


class DispatchTests(unittest.TestCase):
    def setUp(self):
        # Transport is mocked; Python is a portable stand-in, not a live Codex binary.
        self.enterContext(patch('qplatform.research_observer.resolve_cli', return_value=Path(sys.executable)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = "role_test"
        self.policy = load_policy(ROOT / "configs/research_dispatch.json")
        self.script = self.root / "science.py"
        self.script.write_text(
            "import os,json\nfrom pathlib import Path\n"
            "p=Path(os.environ['QPT_OUTPUT_DIR'])\n"
            "(p/'observer_summary.json').write_text(json.dumps({'pnl':-2,'note':'test negative outcome'}))\n")
        self.manifest = {"schema": "research_runtime_v1", "run_id": self.run_id,
                         "bindings": {"science": {"path": str(self.script), "sha256": file_digest(self.script), "role": "entrypoint"}},
                         "stages": [{"id": "negative_case", "uses": ["science"], "phase": "validation",
                                     "command": ["{python}", str(self.script)], "outputs": ["observer_summary.json"],
                                     "resources": {"max_memory_bytes": 256*1024**2, "available_floor_bytes": 16*1024**2,
                                                   "wait_timeout_s": 3, "wall_timeout_s": 10, "stop_grace_s": 1}}]}
        self.path = self.root / "manifest.json"
        atomic_json(self.path, self.manifest)
        self.request = {"manifest": str(self.path), "manifest_sha256": file_digest(self.path),
                        "policy": self.policy, "cli": sys.executable}
        self.events = dispatch_root(self.root, self.run_id)

    def run_runtime(self):
        result = run_manifest(self.root, self.path, poll_s=.05)
        self.assertEqual(result["status"], "execution_complete", result)
        return result

    def fake_observe(self, packet, directory, policy, cli, pause):
        report = report_for(packet)
        atomic_json(directory / "report.json", report)
        atomic_json(directory / "receipt.json", {"packet_sha256": file_digest(directory / "packet.json"),
                                               "report_sha256": file_digest(directory / "report.json")})
        return report

    def test_packet_preserves_negative_result_and_all_stages(self):
        self.run_runtime()
        packet = build_packet(self.root, self.run_id)
        self.assertEqual(packet["stages"][0]["untrusted_summary"]["pnl"], -2)
        self.assertEqual(packet["scientific_validity"], "not_assessed")
        self.assertFalse(packet["research_complete"])
        validate_report(report_for(packet), packet)

    def test_observer_cannot_drop_outcome_change_status_or_issue_verdict(self):
        self.run_runtime()
        packet = build_packet(self.root, self.run_id)
        for mutate in (
            lambda r: r.update(scientific_verdict="valid"),
            lambda r: r.update(stage_checks=[]),
            lambda r: r["stage_checks"][0].update(status="profitable"),
            lambda r: r.update(new_threshold=42),
        ):
            report = report_for(packet)
            mutate(report)
            with self.assertRaises(ValueError):
                validate_report(report, packet)

    def test_running_never_invokes_model(self):
        with patch("qplatform.research_dispatch.observe") as observer:
            with self.assertRaisesRegex(ValueError, "terminal"):
                process_event(self.root, self.run_id, self.request)
        observer.assert_not_called()

    def test_event_dedup_model_once_delivery_once(self):
        self.run_runtime()
        with patch("qplatform.research_dispatch.observe", side_effect=self.fake_observe) as observer, \
                patch("qplatform.research_observer.bounded_cli", return_value={"returncode": 0}) as queue:
            first = process_event(self.root, self.run_id, self.request)
            second = process_event(self.root, self.run_id, self.request)
        self.assertEqual(first["status"], "queued")
        self.assertEqual(second["status"], "queued")
        observer.assert_called_once()
        queue.assert_called_once()
        self.assertIn(f'--workspace "{self.root.resolve()}"', queue.call_args.args[0][-1])
        self.assertEqual(first["target"], self.policy["research_thread_id"])
        with self.assertRaisesRegex(ValueError, "identity"):
            acknowledge_delivery(self.root, self.run_id, first["event_id"], self.policy["platform_thread_id"])
        ack = acknowledge_delivery(self.root, self.run_id, first["event_id"], first["target"])
        self.assertTrue(ack["consumed_by_owner"])
        atomic_json(self.events / "current.json", first)
        status = dispatch_status(self.root, self.run_id)
        self.assertTrue(status["consumed_by_owner"])
        self.assertEqual(status["delivery_status"], "acknowledged")
        ack_path = self.events / "events" / first["event_id"] / "delivery/delivery_ack.json"
        atomic_json(ack_path, {**ack, "queue_receipt_sha256": "tampered"})
        with self.assertRaisesRegex(ValueError, "receipt_mismatch"):
            dispatch_status(self.root, self.run_id)

    def test_observer_failure_routes_platform_without_retry(self):
        self.run_runtime()
        with patch("qplatform.research_dispatch.observe", side_effect=ValueError("route drift")) as observer, \
                patch("qplatform.research_observer.bounded_cli", return_value={"returncode": 0}):
            first = process_event(self.root, self.run_id, self.request)
            second = process_event(self.root, self.run_id, self.request)
        self.assertEqual(first["target"], self.policy["platform_thread_id"])
        self.assertEqual(second["status"], "queued")
        observer.assert_called_once()

    def test_reused_observer_rejects_changed_model_input_without_new_model(self):
        self.run_runtime()

        def observer(packet, directory, *args):
            result = self.fake_observe(packet, directory, *args)
            atomic_json(directory / "observer_input.json", packet)
            receipt = read_json(directory / "receipt.json")
            receipt["observer_input_sha256"] = file_digest(directory / "observer_input.json")
            atomic_json(directory / "receipt.json", receipt)
            return result

        with patch("qplatform.research_dispatch.observe", side_effect=observer), \
                patch("qplatform.research_dispatch.deliver", side_effect=RuntimeError("before queue")):
            with self.assertRaisesRegex(RuntimeError, "before queue"):
                process_event(self.root, self.run_id, self.request)
        event_id = build_packet(self.root, self.run_id)["event_id"]
        path = self.events / "events" / event_id / "observer/observer_input.json"
        atomic_json(path, {"tampered": True})
        with patch("qplatform.research_dispatch.observe") as model, \
                patch("qplatform.research_observer.bounded_cli", return_value={"returncode": 0}):
            result = process_event(self.root, self.run_id, self.request)
        model.assert_not_called()
        self.assertEqual(result["target"], self.policy["platform_thread_id"])
        self.assertEqual(result["observer_error"], "observer_input_receipt_mismatch")

    def test_corrupt_artifact_routes_platform_without_model(self):
        result = self.run_runtime()
        receipt = Path(result["stages"]["negative_case"]["receipt"])
        (receipt.parent / "artifacts/observer_summary.json").write_text("{}")
        with patch("qplatform.research_dispatch.observe") as observer, \
                patch("qplatform.research_observer.bounded_cli", return_value={"returncode": 0}):
            result = process_event(self.root, self.run_id, self.request)
        observer.assert_not_called()
        self.assertEqual(result["target"], self.policy["platform_thread_id"])
        self.assertIn("integrity", result["observer_error"])

    def test_user_pause_suppresses_model_and_delivery(self):
        self.run_runtime()
        stop_dispatch(self.root, self.run_id)
        with patch("qplatform.research_dispatch.observe") as observer, patch("qplatform.research_dispatch.deliver") as queue:
            result = process_event(self.root, self.run_id, self.request)
        self.assertIn("paused", result["status"])
        observer.assert_not_called()
        queue.assert_not_called()

    def test_runtime_stop_marker_also_suppresses_handoff(self):
        self.run_runtime()
        atomic_json(_root(self.root, self.run_id) / "stop.json", {"reason": "budget"})
        result = process_event(self.root, self.run_id, self.request)
        self.assertIn("paused", result["status"])

    def test_pause_during_observer_suppresses_delivery(self):
        self.run_runtime()
        def observer(*args):
            result = self.fake_observe(*args)
            stop_dispatch(self.root, self.run_id)
            return result
        with patch("qplatform.research_dispatch.observe", side_effect=observer), patch("qplatform.research_dispatch.deliver") as queue:
            result = process_event(self.root, self.run_id, self.request)
        self.assertEqual(result["status"], "paused_no_delivery")
        queue.assert_not_called()

    def test_ambiguous_queue_never_resubmits(self):
        self.run_runtime()
        packet = build_packet(self.root, self.run_id)
        directory = self.events / "delivery_test"
        with patch("qplatform.research_observer.bounded_cli", side_effect=RuntimeError("timeout")) as queue:
            with self.assertRaises(RuntimeError):
                deliver(packet, None, directory, self.policy, Path(sys.executable), self.events / "pause.json")
            with self.assertRaisesRegex(ValueError, "uncertain"):
                deliver(packet, None, directory, self.policy, Path(sys.executable), self.events / "pause.json")
        queue.assert_called_once()

    def test_supervisor_runs_existing_engine_and_handoff(self):
        atomic_json(self.events / "request.json", self.request)
        with patch("qplatform.research_dispatch.observe", side_effect=self.fake_observe), \
                patch("qplatform.research_observer.bounded_cli", return_value={"returncode": 0}):
            result = supervise(self.root, self.run_id)
        self.assertEqual(result["status"], "queued")
        self.assertEqual(dispatch_status(self.root, self.run_id)["status"], "queued")

    def test_detached_start_does_not_allow_wrong_run_or_duplicate(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            start_dispatch(self.root, "wrong", self.path, ROOT / "configs/research_dispatch.json")
        atomic_json(self.events / "current.json", {"status": "running", "controller": {"pid": os.getpid(),
                    "create_time": __import__("psutil").Process().create_time()}})
        with patch("qplatform.research_dispatch.resolve_cli", return_value=Path(sys.executable)):
            with self.assertRaisesRegex(ValueError, "active"):
                start_dispatch(self.root, self.run_id, self.path, ROOT / "configs/research_dispatch.json")

    def test_bounded_cli_actual_process_exit_timeout_and_pause(self):
        result = bounded_cli([sys.executable, "-c", "print('test')"], self.root / "cli_ok",
                             timeout_s=5, max_memory_bytes=256*1024**2)
        self.assertEqual(result["returncode"], 0)
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            bounded_cli([sys.executable, "-c", "import time; time.sleep(30)"], self.root / "cli_timeout",
                        timeout_s=.2, max_memory_bytes=256*1024**2)
        marker = self.root / "pause.json"
        marker.touch()
        with self.assertRaisesRegex(RuntimeError, "paused"):
            bounded_cli([sys.executable, "-c", "raise Exception('must not start')"], self.root / "cli_paused",
                        timeout_s=5, max_memory_bytes=256*1024**2, pause=StopSignals(marker))


if __name__ == "__main__":
    unittest.main()
