import copy
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qplatform.control_io import atomic_json
from qplatform.research_delivery import acknowledgment_deadline, delivery_health
from qplatform.research_dispatch import dispatch_status, load_policy, validate_policy
from qplatform.research_observer import StopSignals, acknowledge_delivery, bounded_cli, deliver, receive_delivery
from qplatform.research_readiness import file_digest, read_json
from qplatform.research_watchdog import register, tick


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        # Transport is mocked; Python is a portable stand-in, not a live Codex binary.
        self.enterContext(patch('qplatform.research_observer.resolve_cli', return_value=Path(sys.executable)))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = 'delivery_test'
        self.event_id = 'a' * 64
        self.base = self.root / 'workspace_out/runs' / self.run / 'operations/research_dispatch'
        self.event = self.base / 'events' / self.event_id
        self.directory = self.event / 'delivery'
        self.pause = StopSignals(self.base / 'pause.json')
        self.policy = load_policy(ROOT / 'configs/research_dispatch.json')
        self.enterContext(patch('qplatform.research_watchdog.resolve_cli', return_value=Path(sys.executable)))
        self.now = datetime.now(timezone.utc)

    def receipt(self, age=0):
        queued = (self.now - timedelta(seconds=age)).isoformat()
        r = {'status': 'queued', 'target': self.policy['platform_thread_id'], 'event_id': self.event_id,
             'queued_at': queued, 'ack_timeout_s': 120, 'ack_due_at': acknowledgment_deadline(queued, 120)}
        atomic_json(self.directory / 'delivery_receipt.json', r)
        atomic_json(self.base / 'current.json', r)
        register(self.root, self.run, request={'policy': self.policy, 'cli': 'fake'})
        return r

    def test_deadline_pending_boundary_and_overdue(self):
        self.receipt(120)
        self.assertEqual(delivery_health(self.directory, now=self.now)['status'], 'awaiting_ack')
        late = delivery_health(self.directory, now=self.now + timedelta(microseconds=1))
        self.assertEqual(late['status'], 'acknowledgment_overdue')
        self.assertFalse(late['consumed_by_owner'])
        self.assertIsNone(late['transport_arrival_at'])

    def test_ack_is_verified_and_late_ack_keeps_missed_deadline(self):
        self.receipt(200)
        acknowledge_delivery(self.root, self.run, self.event_id, self.policy['platform_thread_id'])
        health = delivery_health(self.directory)
        self.assertEqual(health['status'], 'acknowledged_late')
        self.assertFalse(health['within_ack_slo'])
        self.assertTrue(health['consumed_by_owner'])
        self.assertEqual(dispatch_status(self.root, self.run)['delivery_status'], 'acknowledged_late')

    def test_timely_ack_and_tamper(self):
        self.receipt()
        acknowledge_delivery(self.root, self.run, self.event_id, self.policy['platform_thread_id'])
        self.assertTrue(delivery_health(self.directory)['within_ack_slo'])
        ack = self.directory / 'delivery_ack.json'
        data = read_json(ack)
        atomic_json(ack, {**data, 'queue_receipt_sha256': 'tampered'})
        with self.assertRaisesRegex(ValueError, 'receipt_mismatch'):
            delivery_health(self.directory)
        with self.assertRaisesRegex(ValueError, 'receipt_mismatch'):
            acknowledge_delivery(self.root, self.run, self.event_id, self.policy['platform_thread_id'])

    def test_receive_and_repeated_ack_do_not_reopen_old_test(self):
        self.receipt()
        target = self.policy['platform_thread_id']
        pending = receive_delivery(self.root, self.run, self.event_id, target)
        self.assertTrue(pending['ack_required'])
        first = acknowledge_delivery(self.root, self.run, self.event_id, target)
        self.assertEqual(first['disposition'], 'acknowledged_now')
        ack_path = self.directory / 'delivery_ack.json'
        original = ack_path.read_bytes(), ack_path.stat().st_mtime_ns
        for _ in range(3):
            received = receive_delivery(self.root, self.run, self.event_id, target)
            repeated = acknowledge_delivery(self.root, self.run, self.event_id, target)
            self.assertEqual(received['disposition'], 'already_consumed')
            self.assertEqual(repeated['disposition'], 'already_consumed')
            self.assertFalse(repeated['ack_required'])
            self.assertFalse(repeated['research_authority'])
            self.assertIn('pause_has_priority', repeated['continuation'])
            self.assertEqual((ack_path.read_bytes(), ack_path.stat().st_mtime_ns), original)
        with self.assertRaisesRegex(ValueError, 'identity'):
            receive_delivery(self.root, self.run, self.event_id, 'wrong_thread')

    def test_native_receive_skips_ack_without_requiring_old_packet(self):
        self.receipt()
        target = self.policy['platform_thread_id']
        acknowledge_delivery(self.root, self.run, self.event_id, target)
        process = subprocess.run([sys.executable, str(ROOT / 'scripts/qpt.py'), 'research-dispatch',
            'receive', '--workspace', str(self.root), '--run-id', self.run, '--event-id', self.event_id,
            '--thread-id', target], cwd=ROOT, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result['disposition'], 'already_consumed')
        self.assertFalse(result['ack_required'])

    def test_legacy_receipt_gets_explicit_default_deadline(self):
        receipt = self.receipt(121)
        receipt.pop('ack_timeout_s'); receipt.pop('ack_due_at')
        atomic_json(self.directory / 'delivery_receipt.json', receipt)
        self.assertEqual(delivery_health(self.directory, now=self.now)['status'], 'acknowledgment_overdue')

    def test_invalid_timestamps_and_policy_fail_closed(self):
        receipt = self.receipt()
        for values in ({'queued_at': '2026-10-28T00:00:00'}, {'ack_timeout_s': True},
                       {'ack_due_at': '2099-01-01T00:00:00+00:00'}):
            atomic_json(self.directory / 'delivery_receipt.json', {**receipt, **values})
            with self.assertRaises(ValueError):
                delivery_health(self.directory)
        for value in (True, -1, 0, 3601, '120'):
            policy = copy.deepcopy(self.policy); policy['delivery']['ack_timeout_s'] = value
            with self.assertRaises(ValueError):
                validate_policy(policy)

    def test_watchdog_deadline_is_visible_without_resending_queue(self):
        self.receipt(121)
        with patch('qplatform.research_watchdog.deliver') as queue:
            first = tick(self.root, submit=True)
            second = tick(self.root, submit=True)
        queue.assert_not_called()
        self.assertEqual(first['attention_count'], 1)
        self.assertEqual(second['runs'][0]['blocker'], 'delivery_acknowledgment_overdue')
        self.assertEqual(dispatch_status(self.root, self.run)['status'], 'at_risk')

    def test_ack_closes_active_deadline_alert_but_preserves_lateness(self):
        self.receipt(121)
        tick(self.root, submit=True)
        acknowledge_delivery(self.root, self.run, self.event_id, self.policy['platform_thread_id'])
        result = tick(self.root, submit=True)
        self.assertEqual(result['attention_count'], 0)
        self.assertEqual(result['runs'][0]['delivery_checks'][0]['health']['status'], 'acknowledged_late')

    def test_pause_suppresses_overdue_alert(self):
        self.receipt(121)
        atomic_json(self.base / 'pause.json', {'reason': 'operator'})
        result = tick(self.root, submit=True)
        self.assertEqual(result['attention_count'], 0)

    def test_dry_run_never_queues_notification(self):
        self.receipt(121)
        with patch('qplatform.research_watchdog.deliver') as queue:
            self.assertEqual(tick(self.root)['attention_count'], 1)
        queue.assert_not_called()

    def test_corrupt_ack_and_receipt_are_not_green(self):
        self.receipt()
        atomic_json(self.directory / 'delivery_ack.json', {'event_id': 'other'})
        self.assertEqual(tick(self.root)['attention_count'], 1)
        (self.directory / 'delivery_receipt.json').write_text('{broken')
        self.assertEqual(tick(self.root)['attention_count'], 1)

    def test_missing_queue_receipt_is_fault_not_silent_green(self):
        self.receipt()
        (self.directory / 'delivery_receipt.json').unlink()
        result = tick(self.root)
        self.assertEqual(result['attention_count'], 1)
        self.assertEqual(result['runs'][0]['blocker'], 'queued_delivery_receipt_missing')
        self.assertEqual(dispatch_status(self.root, self.run)['status'], 'at_risk')

    def test_queue_failure_remains_durable_without_retry(self):
        register(self.root, self.run, request={'policy': self.policy, 'cli': 'fake'})
        atomic_json(self.base / 'current.json', {'status': 123})
        with patch('qplatform.research_observer.bounded_cli', side_effect=RuntimeError('capacity')) as queue:
            first = tick(self.root, submit=True)
            second = tick(self.root, submit=True)
        queue.assert_called_once()
        self.assertEqual(first['attention_count'], 1)
        self.assertIn('uncertain', second['runs'][0]['delivery'])

    def test_queue_message_id_and_deadline_recorded(self):
        packet = {'event_id': self.event_id, 'run_id': self.run, 'workspace': str(self.root),
                  'runtime_status': 'execution_complete'}
        self.directory.mkdir(parents=True)
        msg_id = '01a182b8-4c0b-7862-a7f8-3ed382d3adc2'
        (self.directory / 'stdout.jsonl').write_text(f'Queued message {msg_id} for thread x.')
        with patch('qplatform.research_observer.bounded_cli', return_value={'returncode': 0}) as queue:
            receipt = deliver(packet, None, self.directory, self.policy, Path(sys.executable), self.pause)
        self.assertEqual(receipt['queue_message_id'], msg_id)
        self.assertEqual(receipt['ack_timeout_s'], 120)
        self.assertEqual(queue.call_args.kwargs['cwd'], ROOT)
        self.assertIn('already_consumed', queue.call_args.args[0][-1])
        self.assertEqual(receipt['instruction_scope'], 'single_event')

    def test_native_status_and_watchdog_report_overdue_nonzero(self):
        self.receipt(121)
        for command in ([sys.executable, str(ROOT / 'scripts/qpt.py'), 'research-dispatch', 'status',
                         '--workspace', str(self.root), '--run-id', self.run],
                        [sys.executable, '-m', 'qplatform.research_watchdog', '--workspace', str(self.root)]):
            process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 2, process.stderr.decode(errors='replace'))
            self.assertIn('overdue', process.stdout.decode())

    def test_short_process_cwd_with_long_evidence_directory(self):
        directory = self.root / ('a'*100) / ('b'*100) / ('c'*50)
        result = bounded_cli([sys.executable, '-c', 'print("ok")'], directory, cwd=self.root,
                             timeout_s=5, max_memory_bytes=256*1024**2)
        self.assertEqual(result['returncode'], 0)
        self.assertIn('ok', (directory / 'stdout.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
