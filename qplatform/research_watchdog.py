"""Model-free liveness monitor for explicitly registered dispatches only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .control_io import atomic_json
from .research_readiness import read_json
from .research_runtime import _alive, _digest, _lock, _now, _root, runtime_status
from .research_observer import StopSignals, deliver, resolve_cli
from .research_delivery import delivery_health


def register(workspace: Path, run_id: str, *, request: dict | None = None):
    _root(workspace, run_id)
    directory = workspace / "workspace_out/state/research_watchdog/registrations"
    record = {"run_id": run_id, "registered_at": _now()}
    if request is not None:
        from .research_dispatch import validate_policy
        record["fault_delivery"] = {"policy": validate_policy(request["policy"]),
                                    "cli": request["cli"]}
    atomic_json(directory / f"{run_id}.json", record)


def _fault_route(workspace: Path, base: Path, registration: Path) -> tuple[dict, Path, str]:
    from .research_dispatch import load_policy, validate_policy
    errors = []
    for source, path in (("request", base / "request.json"), ("registration", registration)):
        try:
            context = read_json(path)
            if source == "registration":
                context = context["fault_delivery"]
            policy = validate_policy(context["policy"])
            if not isinstance(context["cli"], str) or not context["cli"].strip():
                raise ValueError("missing notification CLI")
            return policy, resolve_cli(context["cli"]), source
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{source}:{type(exc).__name__}:{exc}")
    try:
        # Old registrations have no route copy. Only the canonical policy may
        # supply that fallback; a corrupt run never gets to choose a recipient.
        return load_policy(workspace / "configs/research_dispatch.json"), resolve_cli(), "canonical_policy"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("fault_delivery_route_unavailable:" + ";".join(errors) + ";" + str(exc)) from exc


def _publish_fault(workspace: Path, run_id: str, registration: Path, row: dict, submit: bool):
    base = _root(workspace, run_id).parent / "research_dispatch"
    pause = StopSignals(base / "pause.json", _root(workspace, run_id) / "stop.json")
    if pause.exists():
        row.update(status="paused", blocker=None)
        return
    event_id = _digest({"run_id": run_id, "controller": row.get("controller"), "blocker": row["blocker"]})
    event = base / "events" / event_id
    packet = {"schema": "research_watchdog_fault_v1", "event_id": event_id,
              "run_id": run_id, "workspace": str(workspace), "runtime_status": "engineering_failure",
              "blocker": row["blocker"], "observer_calls": 0, "scientific_validity": "not_assessed"}
    if not (event / "observer/packet.json").exists():
        atomic_json(event / "observer/packet.json", packet)
    delivery = event / "delivery"
    row.update(event_id=event_id, evidence=str(event), delivery="pending")
    if (delivery / "delivery_receipt.json").exists():
        row["delivery"] = "queued"
    elif (delivery / "delivery_claim.json").exists():
        row["delivery"] = "uncertain_manual_reconciliation_required"
    elif submit:
        try:
            policy, cli, source = _fault_route(workspace, base, registration)
            row["routing_source"] = source
            result = deliver(packet, None, delivery, policy, cli, pause,
                             observer_error="no-model watchdog: " + row['blocker'])
            row["delivery"] = result['status']
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            row["delivery"] = "failed_or_uncertain_manual_reconciliation_required"
            row["delivery_error"] = str(exc)
    atomic_json(event / "watchdog.json", row)


def _check_deliveries(workspace: Path, run_id: str, row: dict):
    base = _root(workspace, run_id).parent / "research_dispatch"
    pause = StopSignals(base / "pause.json", _root(workspace, run_id) / "stop.json")
    if pause.exists() or str(row.get("status", "")).startswith("paused"):
        return
    checks = []
    # Only this registered run's explicit outbox, never historical run discovery.
    for path in sorted((base / "events").glob("*/delivery/delivery_receipt.json")):
        event = path.parent.parent
        try:
            health = delivery_health(path.parent)
            if health.get("event_id") != event.name:
                raise ValueError("delivery_event_identity_mismatch")
            blocker = "delivery_acknowledgment_overdue" if health["status"] == "acknowledgment_overdue" else None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            health = {"status": "invalid_delivery_evidence", "error": str(exc)}
            blocker = "invalid_delivery_evidence"
        check = {"event_id": event.name, "health": health, "blocker": blocker}
        if blocker and not row.get("blocker"):
            row.update(status="at_risk", blocker=blocker)
        checks.append(check)
    row["delivery_checks"] = checks


def inspect_dispatch(workspace: Path, run_id: str) -> dict:
    base = _root(workspace, run_id).parent / "research_dispatch"
    pause = StopSignals(base / "pause.json", _root(workspace, run_id) / "stop.json")
    if pause.exists():
        return {"run_id": run_id, "status": "paused", "blocker": None}
    path = base / "current.json"
    if not path.exists():
        return {"run_id": run_id, "status": "at_risk", "blocker": "dispatch_state_missing"}
    current = read_json(path)
    if not isinstance(current, dict) or not isinstance(current.get("status"), str):
        raise ValueError("invalid_dispatch_state")
    status, blocker = current.get("status"), None
    if status not in {'starting', 'running', 'observing', 'dispatch_failed', 'queued', 'paused',
                      'paused_no_delivery', 'paused_no_observer_or_delivery', 'acceptance_failed_no_delivery'}:
        raise ValueError("unknown_dispatch_state:" + status)
    if status in {"starting", "running", "observing"}:
        if not _alive(current.get("controller", {})):
            blocker = "dispatch_controller_lost_no_automatic_restart"
        elif status == "running":
            runtime = runtime_status(workspace, run_id, compact=True)
            if runtime.get("status") in {"at_risk", "interrupted"}:
                blocker = runtime.get("blocker", "runtime_liveness_failure")
        elif status == "observing":
            request = read_json(base / "request.json")
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            budget = request['policy']['observer']['timeout_s'] + request['policy']['delivery']['timeout_s'] + 60
            if age > budget:
                blocker = "observer_or_delivery_supervision_stalled"
    elif status == "dispatch_failed":
        blocker = "dispatch_failed:" + str(current.get("blocker", "unknown"))[:500]
    elif status == 'acceptance_failed_no_delivery':
        blocker = 'acceptance_failed_no_delivery'
    elif status == 'queued' and current.get('event_id'):
        if not (base / 'events' / current['event_id'] / 'delivery/delivery_receipt.json').exists():
            blocker = 'queued_delivery_receipt_missing'
    return {"run_id": run_id, "status": "at_risk" if blocker else status,
            "blocker": blocker, "controller": current.get("controller")}


def tick(workspace: Path, *, submit: bool = False) -> dict:
    workspace = workspace.resolve()
    root = workspace / "workspace_out/state/research_watchdog"
    rows = []
    with _lock(root):
        for registration in sorted((root / "registrations").glob("*.json")):
            run_id = registration.stem
            try:
                row = inspect_dispatch(workspace, run_id)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row = {"run_id": run_id, "status": "watchdog_inspection_failure",
                       "blocker": f"watchdog_inspection_failure:{type(exc).__name__}:{exc}"}
            if row.get("blocker"):
                try:
                    _publish_fault(workspace, run_id, registration, row, submit)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    # One unwritable run must not prevent checks of its siblings.
                    row.update(delivery="fault_publication_failed", delivery_error=str(exc))
                    atomic_json(root / "publication_failures" / (_digest(row) + '.json'), row)
            _check_deliveries(workspace, run_id, row)
            rows.append(row)
        result = {"schema": "research_watchdog_status_v1", "checked_at": _now(), "model_calls": 0,
                  "automatic_restarts": 0, "runs": rows,
                  "attention_count": sum(r.get("blocker") is not None for r in rows)}
        result.update(status="at_risk" if result["attention_count"] else
                      ("observed_no_blocker" if rows else "no_registered_dispatches"),
                      updated_at=result["checked_at"])
        atomic_json(root / "current.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    result = tick(args.workspace, submit=args.submit)
    print(json.dumps(result))
    raise SystemExit(2 if result["attention_count"] else 0)


if __name__ == "__main__":
    main()
