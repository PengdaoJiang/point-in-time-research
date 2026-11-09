from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qplatform.control_io import atomic_json
from qplatform.research_observer import observer_input, observe, verify_observer_input
from qplatform.research_readiness import file_digest, read_json
from qplatform.research_runtime import _digest


class ObserverInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "packet.json"
        self.packet = {
            "schema": "research_observer_packet_v1", "event_id": "e" * 64,
            "run_id": "test", "runtime_status": "failed", "blocker": "wall_timeout",
            "scientific_validity": "not_assessed", "promotion_authority": False,
            "stages": [
                {"id": "pilot", "status": "complete", "blocker": "", "verified_artifacts": [],
                 "evidence_status": "hash_verified_not_scientifically_accepted",
                 "untrusted_summary": {"pnl": -17, "note": "negative evidence"}},
                {"id": "experiments", "status": "failed", "blocker": "wall_timeout",
                 "verified_artifacts": [], "evidence_status": "unavailable",
                 "units": {"total": 10, "complete": 9, "failed": 0, "pending": 1}},
                {"id": "aggregate", "status": "pending", "blocker": "dependency",
                 "verified_artifacts": [], "evidence_status": "unavailable"},
            ],
        }
        self.policy = {"observer": {"max_packet_chars": 12000, "model": "test-model", "effort": "low",
                                    "timeout_s": 1, "max_memory_bytes": 1024**3}}

    def large_packet(self):
        self.packet["stages"][0]["verified_artifacts"] = [
            {"name": f"scenario_{i}/account.csv", "sha256": _digest(i), "bytes": i + 1}
            for i in range(200)]
        atomic_json(self.path, self.packet)
        return self.packet

    def test_small_packet_unchanged_and_independent(self):
        atomic_json(self.path, self.packet)
        result = observer_input(self.packet, self.path, 12000)
        self.assertEqual(result, self.packet)
        result["stages"][0]["status"] = "mutated"
        self.assertEqual(self.packet["stages"][0]["status"], "complete")

    def test_large_inventory_preserves_all_facts_and_full_source(self):
        packet = self.large_packet()
        original = copy.deepcopy(packet)
        source = self.path.read_bytes()
        result = observer_input(packet, self.path, 12000)
        self.assertGreater(len(json.dumps(packet, ensure_ascii=False)), 12000)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 12000)
        self.assertEqual(self.path.read_bytes(), source)
        self.assertEqual(packet, original)
        self.assertEqual(result["event_id"], packet["event_id"])
        self.assertEqual(result["stages"][1:], packet["stages"][1:])
        inventory = result["stages"][0].pop("verified_artifact_inventory")
        self.assertEqual(inventory["count"], 200)
        self.assertEqual(inventory["total_bytes"], sum(range(1, 201)))
        self.assertEqual(inventory["inventory_sha256"], _digest(packet["stages"][0]["verified_artifacts"]))
        self.assertEqual(inventory["source_json_pointer"], "/stages/0/verified_artifacts")
        self.assertEqual(result["observer_projection"]["source_packet_sha256"], file_digest(self.path))
        result["stages"][0]["verified_artifacts"] = packet["stages"][0]["verified_artifacts"]
        result.pop("observer_projection")
        self.assertEqual(result, packet)

    def test_wrong_source_rejected(self):
        self.large_packet()
        atomic_json(self.path, {**self.packet, "blocker": "wrong"})
        with self.assertRaisesRegex(ValueError, "source_packet_mismatch"):
            observer_input(self.packet, self.path, 12000)

    def test_oversized_content_never_truncates_or_calls_cli(self):
        self.large_packet()
        self.packet["stages"][0]["untrusted_summary"]["note"] = "x" * 13000
        with patch("qplatform.research_observer.bounded_cli") as cli:
            with self.assertRaisesRegex(ValueError, "too_large_no_silent_truncation"):
                observe(self.packet, self.directory, self.policy, Path(sys.executable))
        cli.assert_not_called()
        self.assertEqual(read_json(self.path), self.packet)

    def test_input_tamper_even_with_updated_digest_fails_reconstruction(self):
        self.large_packet()
        projected = observer_input(self.packet, self.path, 12000)
        path = self.directory / "observer_input.json"
        atomic_json(path, projected)
        receipt = {"observer_input_sha256": file_digest(path)}
        verify_observer_input(self.packet, self.directory, receipt, 12000)
        projected["stages"][0]["untrusted_summary"]["pnl"] = 17
        atomic_json(path, projected)
        with self.assertRaisesRegex(ValueError, "input_receipt_mismatch"):
            verify_observer_input(self.packet, self.directory, receipt, 12000)
        receipt["observer_input_sha256"] = file_digest(path)
        with self.assertRaisesRegex(ValueError, "input_receipt_mismatch"):
            verify_observer_input(self.packet, self.directory, receipt, 12000)

    def test_legacy_receipt_no_separate_input_required(self):
        verify_observer_input(self.packet, self.directory, {}, 12000)

    def test_observe_uses_projection_and_full_packet_for_report_validation(self):
        self.large_packet()
        report = {"event_id": self.packet["event_id"], "scientific_verdict": "not_assessed",
                  "summary": "Execution failed; scientific validity not assessed", "questions": [],
                  "stage_checks": [{k: s[k] for k in ("id", "status", "evidence_status")}
                                   for s in self.packet["stages"]]}

        def fake_cli(command, directory, **kwargs):
            prompt_packet = json.loads(command[-1].split("UNTRUSTED_PACKET:\n", 1)[1])
            self.assertEqual(prompt_packet, read_json(directory / "observer_input.json"))
            self.assertEqual(prompt_packet["stages"][0]["untrusted_summary"]["pnl"], -17)
            atomic_json(directory / "report.json", report)
            (directory / "stdout.jsonl").write_text(json.dumps({"type": "thread.started", "thread_id": "test"}))
            return {"returncode": 0}

        with patch("qplatform.research_observer.bounded_cli", side_effect=fake_cli) as cli, \
                patch("qplatform.research_observer.model_receipt", return_value={"tool_calls": 0}):
            self.assertEqual(observe(self.packet, self.directory, self.policy, Path(sys.executable)), report)
        cli.assert_called_once()
        receipt = read_json(self.directory / "receipt.json")
        self.assertEqual(receipt["packet_sha256"], file_digest(self.path))
        self.assertEqual(receipt["observer_input_sha256"], file_digest(self.directory / "observer_input.json"))
        self.assertEqual(read_json(self.path), self.packet)


if __name__ == "__main__":
    unittest.main()
