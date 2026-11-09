"""Event-driven role orchestration around the existing research-runtime."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .control_io import atomic_json
from .research_readiness import file_digest, read_json
from .research_runtime import _alive, _identity, _lock, _now, _root, request_stop, run_manifest, runtime_status, validate_manifest
from .research_observer import StopSignals, build_packet, deliver, observe, resolve_cli, validate_report
from .research_delivery import DEFAULT_ACK_TIMEOUT_S, delivery_health


def dispatch_root(workspace: Path, run_id: str) -> Path:
    return _root(workspace, run_id).parent / "research_dispatch"


def load_policy(path: Path) -> dict:
    return validate_policy(read_json(path))


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("dispatch policy must be an object")
    if policy.get("schema") != "research_dispatch_policy_v1":
        raise ValueError("unsupported dispatch policy")
    observer = policy["observer"]
    if (observer["model"] != "gpt-6-luna" or observer["effort"] != "low"
            or observer["automatic_retry"] is not False):
        raise ValueError("unvalidated observer route; platform qualification required")
    for key in ("timeout_s", "max_packet_chars", "max_memory_bytes"):
        if type(observer[key]) is not int or observer[key] <= 0:
            raise ValueError("observer requires finite positive bounds")
    if observer["max_packet_chars"] > 12000:
        raise ValueError("observer packet exceeds tested command-line budget")
    for key in ("research_thread_id", "platform_thread_id"):
        if not re.fullmatch("[0-9a-f-]{36}", policy[key]):
            raise ValueError("dispatch target requires an exact thread UUID")
    if policy["delivery"]["transport"] != "codex_queue" or policy["delivery"]["model_override"] is not None:
        raise ValueError("delivery must retain the owner's model")
    if type(policy["delivery"]["timeout_s"]) is not int or policy["delivery"]["timeout_s"] <= 0:
        raise ValueError("delivery requires a finite positive timeout")
    timeout = policy["delivery"].get("ack_timeout_s", DEFAULT_ACK_TIMEOUT_S)
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("delivery acknowledgment timeout must be 1..3600 seconds")
    return policy


def process_event(workspace: Path, run_id: str, request: dict, *, purpose: str | None = None) -> dict:
    root = dispatch_root(workspace, run_id)
    pause = StopSignals(root / "pause.json", _root(workspace, run_id) / "stop.json")
    if pause.exists():
        return {"status": "paused_no_observer_or_delivery"}
    packet_error = None
    try:
        packet = build_packet(workspace, run_id)
    except (OSError, ValueError, KeyError) as exc:
        from .research_runtime import _digest
        # A nonterminal runtime is not an event and must never trigger a model.
        if str(exc) == "observer_requires_terminal_event":
            raise
        packet_error = "packet_integrity_failure:" + str(exc)
        packet = {"schema": "research_observer_packet_v1", "run_id": run_id, "workspace": str(workspace.resolve()),
                  "runtime_status": "engineering_failure", "stages": [], "blocker": packet_error,
                  "scientific_validity": "not_assessed", "promotion_authority": False}
        packet["event_id"] = _digest(packet)
    if purpose:
        # Only used by the explicit engineering smoke harness, never by normal dispatch.
        from .research_runtime import _digest
        packet["purpose"] = purpose
        packet["event_id"] = _digest(packet)
    event = root / "events" / packet["event_id"]
    observer_dir = event / "observer"
    delivery_dir = event / "delivery"
    if (delivery_dir / "delivery_receipt.json").exists():
        return read_json(delivery_dir / "delivery_receipt.json")
    observer_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(observer_dir / "packet.json", packet)
    report_path = observer_dir / "report.json"
    observer_error = None
    try:
        if packet_error:
            raise ValueError(packet_error)
        receipt_path = observer_dir / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if file_digest(report_path) != receipt["report_sha256"] or file_digest(observer_dir / "packet.json") != receipt["packet_sha256"]:
                raise ValueError("observer_receipt_mismatch")
            from .research_observer import verify_observer_input
            verify_observer_input(packet, observer_dir, receipt, request["policy"]["observer"]["max_packet_chars"])
            validate_report(read_json(report_path), packet)
        elif (observer_dir / "attempt.json").exists():
            raise ValueError("prior_observer_attempt_incomplete_no_automatic_retry")
        else:
            atomic_json(observer_dir / "attempt.json", {"started_at": _now(), "route": request["policy"]["observer"]})
            observe(packet, observer_dir, request["policy"], Path(request["cli"]), pause)
    except (OSError, ValueError, RuntimeError) as exc:
        observer_error = str(exc)
        atomic_json(observer_dir / "failure.json", {"blocker": observer_error, "time": _now()})
    if pause.exists():
        return {"status": "paused_no_delivery", "event_id": packet["event_id"]}
    if purpose == "orchestration_acceptance_only" and observer_error:
        return {"status": "acceptance_failed_no_delivery", "event_id": packet["event_id"], "observer_error": observer_error}
    result = deliver(packet, report_path if observer_error is None else None, delivery_dir,
                     request["policy"], Path(request["cli"]), pause, observer_error=observer_error)
    return {**result, "observer_error": observer_error, "event_dir": str(event)}


def supervise(workspace: Path, run_id: str, *, resume: bool = False) -> dict:
    root = dispatch_root(workspace, run_id)
    with _lock(root):
        request = read_json(root / "request.json")
        manifest_path = Path(request["manifest"])
        if file_digest(manifest_path) != request["manifest_sha256"]:
            raise ValueError("dispatch_manifest_changed")
        if resume and (root / "pause.json").exists():
            target = root / "pauses" / f"{time.time_ns()}.json"
            target.parent.mkdir(exist_ok=True)
            os.rename(root / "pause.json", target)
        if (root / "pause.json").exists():
            return {"status": "paused"}
        identity = _identity(os.getpid())
        atomic_json(root / "current.json", {"status": "running", "controller": identity, "run_id": run_id, "started_at": _now()})
        try:
            from .execution_fingerprint import verify_manifest_environment
            manifest = read_json(manifest_path)
            verify_manifest_environment(workspace, manifest)
            result = run_manifest(workspace, manifest_path, resume=resume, expected_run_id=run_id)
            verify_manifest_environment(workspace, manifest)
            if result["status"] == "stopped" or (root / "pause.json").exists():
                outcome = {"status": "paused", "run_id": run_id, "observer_calls": 0}
            else:
                atomic_json(root / "current.json", {"status": "observing", "controller": identity, "run_id": run_id})
                outcome = process_event(workspace, run_id, request, purpose=request.get("purpose"))
            atomic_json(root / "current.json", {**outcome, "controller": identity, "run_id": run_id, "finished_at": _now()})
            return outcome
        except BaseException as exc:
            atomic_json(root / "current.json", {"status": "dispatch_failed", "blocker": str(exc),
                                              "controller": identity, "run_id": run_id, "finished_at": _now()})
            raise


def start_dispatch(workspace: Path, run_id: str, manifest_path: Path, policy_path: Path,
                   cli_path: str | None = None, *, resume: bool = False, purpose: str | None = None) -> dict:
    workspace, manifest_path = workspace.resolve(), manifest_path.resolve()
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if manifest["run_id"] != run_id:
        raise ValueError("dispatch_run_identity_mismatch")
    root = dispatch_root(workspace, run_id)
    with _lock(root):
        previous = dispatch_status(workspace, run_id)
        if previous["status"] in {"starting", "running", "observing"}:
            raise ValueError("dispatch_already_active")
        if (root / "request.json").exists():
            if not resume:
                raise ValueError("existing_dispatch_requires_explicit_resume_or_handoff")
            old = read_json(root / "request.json")
            if old["manifest_sha256"] != file_digest(manifest_path):
                raise ValueError("changed_manifest_requires_new_dispatch_run")
            from .execution_fingerprint import verify_manifest_environment
            verify_manifest_environment(workspace, manifest)
        elif runtime_status(workspace, run_id)["status"] != "not_started":
            raise ValueError("existing_runtime_must_be_handed_off_not_relaunched")
        elif purpose != 'orchestration_acceptance_only':
            from .research_storage_policy import validate_storage
            from .execution_fingerprint import verify_manifest_environment
            from .research_readiness import validate_new_pilot_admission
            validate_new_pilot_admission(manifest)
            validate_storage(workspace, manifest)
            verify_manifest_environment(workspace, manifest, required=True)
        policy, cli = load_policy(policy_path), resolve_cli(cli_path)
        request = {"run_id": run_id, "workspace": str(workspace), "manifest": str(manifest_path),
                   "manifest_sha256": file_digest(manifest_path), "policy": policy, "cli": str(cli),
                   "authorized_by": "explicit_research_dispatch_start", "created_at": _now()}
        if purpose is not None:
            if purpose != "orchestration_acceptance_only":
                raise ValueError("unknown dispatch purpose")
            request["purpose"] = purpose
        if not (root / "request.json").exists():
            atomic_json(root / "request.json", request)
        from .research_watchdog import register
        register(workspace, run_id, request=read_json(root / "request.json"))
        command = [sys.executable, "-m", "qplatform.research_dispatch", "--workspace", str(workspace),
                   "--run-id", run_id]
        if resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH"))))
        with (root / "launcher.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                       start_new_session=os.name != "nt")
        atomic_json(root / "current.json", {"status": "starting", "controller": _identity(process.pid),
                                          "run_id": run_id, "started_at": _now()})
    # No model polling: child takes the lock after registration and owns the whole lifecycle.
    return {"status": "dispatched", "pid": process.pid, "run_id": run_id,
            "status_command": f"python scripts/qpt.py research-dispatch status --run-id {run_id}"}


def dispatch_status(workspace: Path, run_id: str) -> dict:
    path = dispatch_root(workspace, run_id) / "current.json"
    result = read_json(path) if path.exists() else {"status": "not_started", "run_id": run_id}
    if result.get("event_id"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(result["event_id"])):
            raise ValueError("invalid_delivery_event_id")
        directory = dispatch_root(workspace, run_id) / "events" / result["event_id"] / "delivery"
        health = delivery_health(directory)
        if (health.get("event_id", result["event_id"]) != result["event_id"]
                or health.get("target", result.get("target")) != result.get("target")):
            raise ValueError("delivery_ack_receipt_mismatch")
        result.update(delivery_health=health, delivery_status=health["status"],
                      consumed_by_owner=health["consumed_by_owner"])
        if health["consumed_by_owner"]:
            result["delivery_ack"] = health["acknowledgment"]
        elif health["status"] == "acknowledgment_overdue":
            result.update(status="at_risk", blocker="delivery_acknowledgment_overdue")
        elif result["status"] == "queued" and health["status"] in {"not_submitted", "submission_uncertain"}:
            result.update(status="at_risk", blocker="queued_delivery_receipt_missing")
    if result["status"] in {"starting", "running", "observing"} and not _alive(result.get("controller", {})):
        return {**result, "status": "interrupted", "blocker": "dispatch_controller_lost_no_automatic_restart"}
    if result["status"] in {"starting", "running", "observing"}:
        result["runtime"] = runtime_status(workspace, run_id, compact=True)
    return result


def stop_dispatch(workspace: Path, run_id: str) -> dict:
    root = dispatch_root(workspace, run_id)
    atomic_json(root / "pause.json", {"requested_at": _now(), "run_id": run_id})
    return request_stop(workspace, run_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Parent owns the registration lock briefly. Child waits without running science.
    root = dispatch_root(args.workspace, args.run_id)
    for _ in range(100):
        try:
            supervise(args.workspace, args.run_id, resume=args.resume)
            return
        except ValueError as exc:
            if str(exc) != "research runtime already owned by another controller":
                raise
            time.sleep(.05)
    raise RuntimeError("dispatch_registration_lock_timeout")


if __name__ == "__main__":
    main()
