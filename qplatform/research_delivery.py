"""Model-independent queue/ack evidence and deadline projection."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .research_readiness import file_digest, read_json

DEFAULT_ACK_TIMEOUT_S = 120


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid_delivery_timestamp")
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None:
        raise ValueError("delivery_timestamp_requires_timezone")
    return instant.astimezone(timezone.utc)


def acknowledgment_deadline(queued_at: str, timeout_s: int) -> str:
    if type(timeout_s) is not int or timeout_s <= 0:
        raise ValueError("delivery_ack_timeout_requires_positive_integer")
    return (timestamp(queued_at) + timedelta(seconds=timeout_s)).isoformat()


def delivery_health(directory: Path, *, now: datetime | None = None) -> dict:
    """Read-only: queue acceptance and human/agent acknowledgment are distinct."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("delivery_timestamp_requires_timezone")
    receipt_path = directory / "delivery_receipt.json"
    if not receipt_path.exists():
        return {"status": "submission_uncertain" if (directory / "delivery_claim.json").exists()
                else "not_submitted", "consumed_by_owner": False}
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("status") != "queued":
        raise ValueError("invalid_queue_receipt")
    queued = timestamp(receipt["queued_at"])
    due_text = acknowledgment_deadline(receipt["queued_at"], receipt.get("ack_timeout_s", DEFAULT_ACK_TIMEOUT_S))
    if receipt.get("ack_due_at", due_text) != due_text:
        raise ValueError("delivery_deadline_mismatch")
    due = timestamp(due_text)
    if now < queued:
        raise ValueError("delivery_clock_regression")
    result = {"status": "acknowledgment_overdue" if now > due else "awaiting_ack",
              "event_id": receipt["event_id"], "target": receipt["target"],
              "queued_at": receipt["queued_at"], "ack_due_at": due_text,
              "elapsed_s": (now - queued).total_seconds(), "consumed_by_owner": False,
              "queue_message_id": receipt.get("queue_message_id"), "transport_arrival_at": None}
    ack_path = directory / "delivery_ack.json"
    if ack_path.exists():
        ack = read_json(ack_path)
        if (not isinstance(ack, dict) or ack.get("event_id") != receipt["event_id"]
                or ack.get("thread_id") != receipt["target"] or ack.get("consumed_by_owner") is not True
                or ack.get("queue_receipt_sha256") != file_digest(receipt_path)):
            raise ValueError("delivery_ack_receipt_mismatch")
        acknowledged = timestamp(ack["acknowledged_at"])
        if acknowledged < queued or acknowledged > now:
            raise ValueError("delivery_ack_clock_invalid")
        result.update(status="acknowledged_late" if acknowledged > due else "acknowledged",
                      consumed_by_owner=True, acknowledgment=ack,
                      queued_to_ack_s=(acknowledged - queued).total_seconds(),
                      within_ack_slo=acknowledged <= due)
    return result


def queue_message_id(directory: Path) -> str | None:
    path = directory / "stdout.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace") as stream:
        output = stream.read(4096)
    match = re.search(r"Queued message ([0-9a-f-]{36}) for thread", output)
    return match[1] if match else None
