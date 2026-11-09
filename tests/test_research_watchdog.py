import sys
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qplatform.control_io import atomic_json
from qplatform.research_watchdog import register, tick
from qplatform.research_dispatch import load_policy
from qplatform.research_runtime import _identity


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch('qplatform.research_watchdog.resolve_cli', return_value=Path(sys.executable)))
        self.base = self.root / "workspace_out/runs/check/operations/research_dispatch"
        policy = load_policy(Path(__file__).resolve().parents[1] / "configs/research_dispatch.json")
        self.request = {"manifest_sha256": "a"*64, "policy": policy, "cli": "fake"}
        atomic_json(self.base / "request.json", self.request)
        register(self.root, "check", request=self.request)
        atomic_json(self.base / "current.json", {"status": "running", "controller": {"pid": -1, "create_time": 0}})

    def test_lost_controller_queue_once_without_observer(self):
        with patch("qplatform.research_watchdog._alive", return_value=False), \
                patch("qplatform.research_observer.bounded_cli", return_value={"returncode": 0}) as cli:
            first = tick(self.root, submit=True)
            second = tick(self.root, submit=True)
        cli.assert_called_once()
        self.assertEqual(first['model_calls'], 0)
        self.assertEqual(first['runs'][0]['event_id'], second['runs'][0]['event_id'])
        self.assertEqual(second['runs'][0]['delivery'], 'queued')

    def test_pause_suppresses_fault_delivery(self):
        atomic_json(self.base / "pause.json", {"reason": "budget"})
        with patch("qplatform.research_watchdog.deliver") as cli:
            result = tick(self.root, submit=True)
        cli.assert_not_called()
        self.assertEqual(result['runs'][0]['status'], 'paused')

    def test_queue_timeout_preserves_outbox_and_never_retries(self):
        with patch("qplatform.research_watchdog._alive", return_value=False), \
                patch("qplatform.research_observer.bounded_cli", side_effect=RuntimeError("timeout")) as cli:
            first = tick(self.root, submit=True)
            second = tick(self.root, submit=True)
        cli.assert_called_once()
        self.assertIn("uncertain", second['runs'][0]['delivery'])
        self.assertTrue((Path(first['runs'][0]['evidence']) / 'observer/packet.json').exists())

    def test_failed_dispatch_and_dry_run(self):
        atomic_json(self.base / "current.json", {"status": "dispatch_failed", "blocker": "disk full"})
        with patch("qplatform.research_watchdog.deliver") as cli:
            result = tick(self.root)
        cli.assert_not_called()
        self.assertEqual(result['attention_count'], 1)
        self.assertIn("disk full", result['runs'][0]['blocker'])

    def test_unregistered_history_not_scanned(self):
        atomic_json(self.root / "workspace_out/runs/old/operations/research_dispatch/current.json", {"status": "dispatch_failed"})
        atomic_json(self.base / "current.json", {"status": "queued"})
        result = tick(self.root)
        self.assertEqual(len(result['runs']), 1)
        self.assertEqual(result['attention_count'], 0)

    def test_native_controller_exit_creates_durable_outbox_without_model(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.2)'])
        identity = _identity(child.pid)
        child.wait(timeout=5)
        atomic_json(self.base / 'current.json', {'status': 'running', 'controller': identity})
        with patch('qplatform.research_watchdog.deliver') as delivery:
            result = tick(self.root, submit=False)
        delivery.assert_not_called()
        self.assertEqual(result['runs'][0]['blocker'], 'dispatch_controller_lost_no_automatic_restart')
        self.assertTrue((Path(result['runs'][0]['evidence']) / 'observer/packet.json').is_file())

    def test_corrupt_state_and_request_notify_once_using_registration(self):
        for name in ('current.json', 'request.json'):
            (self.base / name).write_text('{broken')
        with patch('qplatform.research_observer.bounded_cli', return_value={'returncode': 0}) as cli:
            first = tick(self.root, submit=True)
            second = tick(self.root, submit=True)
        cli.assert_called_once()
        self.assertEqual(first['runs'][0]['routing_source'], 'registration')
        self.assertEqual(first['runs'][0]['status'], 'watchdog_inspection_failure')
        self.assertEqual(first['runs'][0]['event_id'], second['runs'][0]['event_id'])
        self.assertEqual(second['runs'][0]['delivery'], 'queued')
        self.assertEqual(cli.call_args.args[0][3], self.request['policy']['platform_thread_id'])

    def test_legacy_registration_uses_canonical_route(self):
        register(self.root, 'check')
        (self.base / 'request.json').unlink()
        atomic_json(self.root / 'configs/research_dispatch.json', self.request['policy'])
        with patch('qplatform.research_watchdog.resolve_cli', return_value=Path(sys.executable)), \
                patch('qplatform.research_observer.bounded_cli', return_value={'returncode': 0}) as cli:
            result = tick(self.root, submit=True)
        cli.assert_called_once()
        self.assertEqual(result['runs'][0]['routing_source'], 'canonical_policy')

    def test_missing_routing_preserves_fault_outbox_and_failure_health(self):
        register(self.root, 'check')
        (self.base / 'request.json').unlink()
        with patch('qplatform.research_watchdog.deliver') as delivery:
            result = tick(self.root, submit=True)
        delivery.assert_not_called()
        row = result['runs'][0]
        self.assertIn('fault_delivery_route_unavailable', row['delivery_error'])
        self.assertTrue((Path(row['evidence']) / 'observer/packet.json').exists())
        self.assertEqual(result['attention_count'], 1)

    def test_pause_wins_even_with_corrupt_state(self):
        (self.base / 'current.json').write_text('{broken')
        atomic_json(self.base / 'pause.json', {'reason': 'operator'})
        with patch('qplatform.research_watchdog.deliver') as delivery:
            result = tick(self.root, submit=True)
        delivery.assert_not_called()
        self.assertEqual(result['attention_count'], 0)

    def test_cli_fault_health_is_nonzero_without_submission(self):
        process = subprocess.run([sys.executable, '-m', 'qplatform.research_watchdog',
                                  '--workspace', str(self.root)], capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 2, process.stderr.decode())

    def test_unknown_state_is_not_silently_green(self):
        atomic_json(self.base / 'current.json', {'status': 'corrupt-but-valid-json'})
        with patch('qplatform.research_observer.bounded_cli', return_value={'returncode': 0}) as cli:
            result = tick(self.root, submit=True)
        cli.assert_called_once()
        self.assertEqual(result['attention_count'], 1)

    def test_publication_error_does_not_skip_other_registrations(self):
        register(self.root, 'sibling', request=self.request)
        sibling = self.root / 'workspace_out/runs/sibling/operations/research_dispatch'
        atomic_json(sibling / 'current.json', {'status': 'queued'})
        with patch('qplatform.research_watchdog._publish_fault', side_effect=OSError('run unwritable')):
            result = tick(self.root, submit=True)
        self.assertEqual(len(result['runs']), 2)
        self.assertEqual(result['attention_count'], 1)
        self.assertEqual(result['runs'][0]['delivery'], 'fault_publication_failed')


if __name__ == '__main__':
    unittest.main()
