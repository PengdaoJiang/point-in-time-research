"""Bounded, resumable execution of frozen research stages.

Science adapters own features, labels and economics. This module owns launch
gates, resource admission, process custody, receipts and failure visibility.
"""
from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import psutil

from .research_process import ProcessGroup
from .research_readiness import artifact_path, audit_readiness, file_digest, read_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _windows_access_conflict(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in {5, 32, 33} or (
        os.name == "nt" and isinstance(exc, PermissionError) and exc.errno == errno.EACCES
    )


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # A Windows reader may briefly deny replacement of the old snapshot.
        for delay in (.01, .02, .04, .08, .1, None):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if delay is None or not _windows_access_conflict(exc):
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)


def _read_progress(path: Path) -> Any:
    # Atomic replacements can briefly deny readers on Windows. Do not retry
    # malformed content or turn persistent access failures into zero progress.
    for delay in (.01, .02, .04, .08, .1, None):
        try:
            return read_json(path)
        except OSError as exc:
            if delay is None or not _windows_access_conflict(exc):
                raise
            time.sleep(delay)


def _identity(pid: int) -> dict[str, Any]:
    return {"pid": pid, "create_time": psutil.Process(pid).create_time()}


def _alive(identity: dict[str, Any]) -> bool:
    try:
        process = psutil.Process(identity["pid"])
        return abs(process.create_time() - identity["create_time"]) < .01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError, TypeError):
        return False


def _root(workspace: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
        raise ValueError("invalid run_id")
    parent = (workspace / "workspace_out" / "runs").resolve()
    result = (parent / run_id / "operations" / "research_runtime").resolve()
    if not result.is_relative_to(parent):
        raise ValueError("runtime path escapes runs directory")
    return result


@contextlib.contextmanager
def _lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a+b") as stream:
        if not os.fstat(stream.fileno()).st_size:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("research runtime already owned by another controller") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _inputs(workspace: Path, manifest: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in stage["uses"]:
        reference = manifest["bindings"][key]
        path = (workspace / reference["path"]).resolve()
        if not path.is_file() or file_digest(path) != reference["sha256"]:
            raise ValueError(f"input_binding_mismatch:{key}")
        result[key] = reference
    return result


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "research_runtime_v1":
        raise ValueError("unsupported runtime manifest schema")
    _root(Path.cwd(), manifest["run_id"])
    if not manifest.get("stages") or not manifest.get("bindings"):
        raise ValueError("a manifest needs bound inputs and stages")
    for ref in manifest["bindings"].values():
        if ref["role"] not in {"source", "science", "contract", "entrypoint"} or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]):
            raise ValueError("invalid input binding")
    known = {}
    for stage in manifest["stages"]:
        allowed = {"id", "uses", "command", "phase", "depends_on", "pilot_stage", "outputs", "resources", "progress_file", "readiness_contract", "checkpoint_units", "memory_pressure", "unavailable"}
        if stage.keys() - allowed:
            raise ValueError(f"unknown stage fields: {sorted(stage.keys() - allowed)}")
        name = stage["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or name in known:
            raise ValueError("invalid/duplicate stage id")
        if stage["phase"] not in {"pilot", "scale", "validation"}:
            raise ValueError("invalid stage phase")
        if not isinstance(stage["command"], list) or not stage["command"] or not all(isinstance(x, str) and x for x in stage["command"]):
            raise ValueError("command must be a nonempty argv array (no shell)")
        if not isinstance(stage["uses"], list) or not stage["uses"] or not set(stage["uses"]).issubset(manifest["bindings"]):
            raise ValueError("stage must declare known input bindings")
        if not set(stage.get("depends_on", [])).issubset(known):
            raise ValueError("dependencies must appear earlier in execution order")
        if not isinstance(stage["outputs"], list) or not stage["outputs"]:
            raise ValueError("successful exit without declared artifacts is not completion")
        for output in stage["outputs"]:
            artifact_path(Path.cwd(), output)
        resources = stage["resources"]
        expected = {"max_memory_bytes", "available_floor_bytes", "wait_timeout_s", "wall_timeout_s", "stop_grace_s"}
        if resources.keys() != expected:
            raise ValueError("resources must describe only this active stage, with all finite bounds")
        for key, value in resources.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid resource bound: {key}")
        if type(stage.get("checkpoint_units", False)) is not bool:
            raise ValueError("checkpoint_units must be boolean")
        _memory_policy(stage)
        unavailable = stage.get("unavailable")
        if unavailable is not None:
            if (set(unavailable) != {"reason", "evidence_binding"}
                    or not isinstance(unavailable["reason"], str) or not unavailable["reason"].strip()
                    or unavailable["evidence_binding"] not in stage["uses"]
                    or manifest["bindings"][unavailable["evidence_binding"]]["role"] not in {"contract", "source"}):
                raise ValueError("unavailable stage requires a reason and bound capability evidence")
        if stage.get("progress_file"):
            artifact_path(Path.cwd(), stage["progress_file"])
        if stage["phase"] == "pilot":
            scenarios = stage["readiness_contract"]["scenarios"]
            if not scenarios:
                raise ValueError("pilot requires scenario-level evidence")
            for spec in scenarios.values():
                if spec["mode"] not in {"fit", "replay"} or not isinstance(spec["maturity_by_window"], dict):
                    raise ValueError("invalid pilot scenario contract")
                if spec["mode"] == "fit" and not spec.get("required_training_roles"):
                    raise ValueError("fit pilot must declare required training roles, not just row count")
                roles = spec.get("required_training_roles", [])
                if not isinstance(roles, list) or not all(isinstance(x, str) and x.strip() for x in roles):
                    raise ValueError("required training roles must be a list of nonempty strings")
                if not all(type(n) is int and n >= 0 for n in spec["maturity_by_window"].values()):
                    raise ValueError("maturity counts must be nonnegative integers")
        elif stage["phase"] == "scale":
            pilot_id = stage["pilot_stage"]
            if pilot_id not in stage.get("depends_on", []) or known[pilot_id]["phase"] != "pilot":
                raise ValueError("scale requires an explicit successful pilot dependency")
            pilot = known[pilot_id]
            science = lambda row: {key for key in row["uses"] if manifest["bindings"][key]["role"] != "entrypoint"}
            if science(pilot) != science(stage):
                raise ValueError("pilot and scale must bind identical scientific inputs and code")
        known[name] = stage
    if manifest.get("failure_policy", "fail_fast") not in {"fail_fast", "continue_independent"}:
        raise ValueError("invalid failure policy")


def _memory_policy(stage: dict[str, Any]) -> dict[str, float]:
    floor = stage["resources"]["available_floor_bytes"]
    policy = stage.get("memory_pressure", {"grace_s": min(5.0, stage["resources"]["wall_timeout_s"]),
                                           "critical_available_bytes": min(floor, 1024**3)})
    if set(policy) != {"grace_s", "critical_available_bytes"}:
        raise ValueError("memory_pressure requires grace_s and critical_available_bytes")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in policy.values()):
        raise ValueError("invalid memory pressure bound")
    if policy["critical_available_bytes"] > floor or policy["grace_s"] > stage["resources"]["wall_timeout_s"]:
        raise ValueError("memory pressure bounds exceed stage bounds")
    return policy


def _event(root: Path, payload: dict[str, Any]) -> str:
    key = _digest(payload)
    path = root / "outbox" / f"{key}.json"
    if not path.exists():
        _atomic(path, {**payload, "event_id": key, "created_at": _now(), "delivered": False})
    return key


def runtime_status(workspace: Path, run_id: str, *, compact: bool = False) -> dict[str, Any]:
    root = _root(workspace, run_id)
    path = root / "current.json"
    result = read_json(path) if path.exists() else {"run_id": run_id, "status": "not_started"}
    if result.get("run_id") != run_id:
        return {"run_id": run_id, "status": "invalid_state", "blocker": "current_run_identity_mismatch",
                "execution_complete": False, "research_complete": False, "promotion_authority": False}
    if result["status"] in {"running", "waiting_resources", "stopping"}:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(result["heartbeat_at"])).total_seconds()
        if not _alive(result.get("controller", {})):
            result = {**result, "status": "interrupted", "blocker": "controller_lost", "execution_complete": False}
        elif age > 30:
            result = {**result, "status": "at_risk", "blocker": "controller_heartbeat_stale", "heartbeat_age_s": age}
    result["pending_events"] = [str(path) for path in sorted((root / "outbox").glob("*.json")) if not (root / "acks" / path.name).exists()]
    result["promotion_authority"] = False
    # A stage receipt does not certify the full research mandate. Never import
    # the round-local, manually accumulated precedence/error fields here.
    result["research_complete"] = False
    result["status_source"] = str(path)
    result["blocking_stages"] = [{"stage": name, **row} for name, row in result.get("stage_outcomes", {}).items()
                                 if row["status"] in {"failed", "blocked", "unavailable", "stopped"}]
    if compact:
        fields = ("run_id", "status", "manifest_sha256", "stage", "attempt", "heartbeat_at", "heartbeat_age_s",
                  "completed_units", "elapsed_s", "peak_rss_bytes", "available_bytes", "memory_pressure_s",
                  "blocker", "blocking_stages", "stop_reason", "stage_counts", "execution_complete", "execution_settled",
                  "research_complete", "promotion_authority", "status_source")
        return {**{key: result.get(key) for key in fields}, "pending_event_count": len(result["pending_events"])}
    return result


def acknowledge_event(workspace: Path, run_id: str, event_id: str, channel: str) -> dict[str, Any]:
    root = _root(workspace, run_id)
    if not re.fullmatch(r"[0-9a-f]{64}", event_id) or not channel.strip():
        raise ValueError("event id and delivery channel required")
    event = root / "outbox" / f"{event_id}.json"
    if not event.exists():
        raise ValueError("unknown event")
    receipt = root / "acks" / event.name
    if not receipt.exists():
        _atomic(receipt, {"event_id": event_id, "channel": channel, "acknowledged_at": _now(), "event_sha256": file_digest(event)})
    return read_json(receipt)


def request_stop(workspace: Path, run_id: str) -> dict[str, Any]:
    result = {"requested_at": _now(), "mode": "next_safe_boundary", "run_id": run_id}
    _atomic(_root(workspace, run_id) / "stop.json", result)
    return result


def _run_child(
    stage: dict[str, Any], workspace: Path, attempt: Path, binding: str,
    root: Path, heartbeat: Any, poll_s: float, work_binding: str,
) -> dict[str, Any]:
    outputs = attempt / "artifacts"
    outputs.mkdir()
    resources = stage["resources"]
    environment = os.environ.copy()
    code_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (code_root, environment.get("PYTHONPATH"))))
    environment.update(QPT_OUTPUT_DIR=str(outputs), QPT_BINDING_ID=binding,
                       QPT_STOP_FILE=str(attempt / "stop_requested.json"), PYTHONDONTWRITEBYTECODE="1")
    # Do not accidentally inherit another run's unit namespace.
    environment.pop("QPT_CHECKPOINT_DIR", None)
    environment.pop("QPT_WORK_BINDING_ID", None)
    if stage.get("checkpoint_units"):
        environment.update(QPT_CHECKPOINT_DIR=str(root / "checkpoints" / stage["id"]),
                           QPT_WORK_BINDING_ID=work_binding)
    command = [arg.replace("{python}", sys.executable).replace("{output_dir}", str(outputs)) for arg in stage["command"]]
    start = time.monotonic()
    peak_rss = 0
    minimum_available = psutil.virtual_memory().available
    observed: dict[int, dict[str, Any]] = {}
    blocker = None
    stop_at = None
    stop_reason = None
    last_progress = None
    progress_error = None
    process = None
    group = None
    memory_policy = _memory_policy(stage)
    pressure_since = None
    peak_pressure_s = 0.0

    def read_progress() -> None:
        nonlocal last_progress
        if stage.get("progress_file"):
            path = artifact_path(outputs, stage["progress_file"])
            if path.exists():
                units = _read_progress(path)["completed_units"]
                if type(units) is not int or units < 0 or (last_progress is not None and units < last_progress):
                    raise ValueError("nonmonotonic_progress")
                last_progress = units

    try:
        with (attempt / "stdout.log").open("wb") as out, (attempt / "stderr.log").open("wb") as err:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("research_process.py"))], cwd=workspace,
                env=environment, stdin=subprocess.PIPE, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            identity = _identity(process.pid)
            observed[process.pid] = identity
            group = ProcessGroup(process.pid, int(resources["max_memory_bytes"]))
            _atomic(attempt / "process.json", {"owner": _identity(os.getpid()), "child": identity, "command": command})
            process.stdin.write((json.dumps(command) + "\n").encode())
            process.stdin.flush()
            while process.poll() is None:
                elapsed = time.monotonic() - start
                rss = 0
                try:
                    children = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                    for child in children:
                        try:
                            observed[child.pid] = {"pid": child.pid, "create_time": child.create_time()}
                            rss += child.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.NoSuchProcess:
                    pass
                peak_rss = max(peak_rss, rss)
                available = psutil.virtual_memory().available
                minimum_available = min(minimum_available, available)
                if rss > resources["max_memory_bytes"]:
                    blocker = "process_tree_memory_limit"
                elif available < memory_policy["critical_available_bytes"]:
                    blocker = "host_memory_critical"
                if available < resources["available_floor_bytes"]:
                    if pressure_since is None:
                        pressure_since = time.monotonic()
                    pressure_s = time.monotonic() - pressure_since
                    peak_pressure_s = max(peak_pressure_s, pressure_s)
                else:
                    pressure_since, pressure_s = None, 0.0
                if stage.get("progress_file") and blocker is None:
                    progress_path = artifact_path(outputs, stage["progress_file"])
                    try:
                        read_progress()
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        progress_error = {"exception_type": type(exc).__name__, "message": str(exc),
                                          "path": str(progress_path), "errno": getattr(exc, "errno", None),
                                          "winerror": getattr(exc, "winerror", None)}
                        blocker = str(exc) if str(exc) == "nonmonotonic_progress" else "progress_read_failed:" + type(exc).__name__
                if (root / "stop.json").exists() and stop_reason != "safe_stop_requested":
                    stop_at = stop_at if stop_at is not None else time.monotonic()
                    stop_reason = "safe_stop_requested"
                    _atomic(attempt / "stop_requested.json", read_json(root / "stop.json"))
                elif elapsed > resources["wall_timeout_s"] and stop_at is None:
                    # Give the adapter the same checkpoint boundary as a user stop.
                    stop_at = time.monotonic()
                    stop_reason = "wall_timeout"
                    _atomic(attempt / "stop_requested.json", {
                        "requested_at": _now(), "mode": "next_safe_boundary",
                        "reason": stop_reason,
                    })
                elif pressure_s >= memory_policy["grace_s"] and stop_at is None and blocker is None:
                    stop_at, stop_reason = time.monotonic(), "host_memory_pressure"
                    _atomic(attempt / "stop_requested.json", {
                        "requested_at": _now(), "mode": "next_safe_boundary", "reason": stop_reason,
                    })
                if stop_at is not None and time.monotonic() - stop_at > resources["stop_grace_s"]:
                    blocker = blocker or ("stop_grace_exceeded" if stop_reason == "safe_stop_requested" else stop_reason)
                heartbeat("stopping" if stop_at is not None else "running", elapsed_s=round(elapsed, 3),
                          peak_rss_bytes=peak_rss, available_bytes=available, completed_units=last_progress,
                          stop_reason=stop_reason, memory_pressure_s=round(pressure_s, 3))
                if blocker:
                    break
                time.sleep(poll_s)
            returncode = process.poll()
            if stop_at is not None and blocker is None:
                blocker = stop_reason
            if blocker is None and returncode != 0:
                blocker = f"child_exit:{returncode}"
            if blocker is None:
                try:
                    read_progress()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    blocker = "final_progress_invalid:" + str(exc)
    finally:
        if group is not None:
            group.close()
        if process is not None:
            if group is None and process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            process.stdin.close()
        deadline = time.monotonic() + 10
        while any(_alive(x) for x in observed.values()) and time.monotonic() < deadline:
            time.sleep(.05)
    drained = not any(_alive(x) for x in observed.values())
    return {"returncode": returncode, "blocker": blocker or (None if drained else "owned_processes_not_drained"),
            "elapsed_s": time.monotonic() - start, "peak_rss_bytes": peak_rss,
            "minimum_available_bytes": minimum_available, "owned_processes_drained": drained,
            "completed_units": last_progress, "stop_reason": stop_reason, "progress_error": progress_error,
            "peak_memory_pressure_s": peak_pressure_s}


def _reusable(root: Path, stage: dict[str, Any], binding: str) -> dict[str, Any] | None:
    pointer = root / "stages" / stage["id"] / "latest_success.json"
    if not pointer.exists():
        return None
    try:
        ref = read_json(pointer)
        receipt_path = artifact_path(root, ref["receipt"])
        if file_digest(receipt_path) != ref["sha256"]:
            return None
        receipt = read_json(receipt_path)
        if receipt["binding_id"] != binding or receipt["status"] != "succeeded":
            return None
        directory = receipt_path.parent / "artifacts"
        if any(file_digest(artifact_path(directory, name)) != digest for name, digest in receipt["artifacts"].items()):
            return None
        if stage["phase"] == "pilot" and not audit_readiness(directory, stage["readiness_contract"], binding)["ready"]:
            return None
        return {"receipt": str(receipt_path), "sha256": ref["sha256"], "binding_id": binding, "reused": True}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_manifest(workspace: Path, manifest_path: Path, *, poll_s: float = 1.0, resume: bool = False,
                 expected_run_id: str | None = None) -> dict[str, Any]:
    if not math.isfinite(poll_s) or not 0 < poll_s <= 5:
        raise ValueError("poll_s must be finite and in (0, 5]")
    workspace = workspace.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    validate_manifest(manifest)
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("manifest_run_identity_mismatch")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    root = _root(workspace, manifest["run_id"])
    owner = _identity(os.getpid())
    platform_binding = {name: file_digest(Path(__file__).with_name(name)) for name in (
        "research_runtime.py", "research_readiness.py", "research_process.py", "research_checkpoints.py")}
    platform_binding.update(python=sys.version, executable=sys.executable, psutil=psutil.__version__)
    completed: dict[str, Any] = {}
    outcomes: dict[str, Any] = {stage["id"]: {"status": "pending"} for stage in manifest["stages"]}
    current_stage = None
    attempt = None
    latest_process: dict[str, Any] = {}

    def heartbeat(status: str, **extra: Any) -> None:
        _atomic(root / "current.json", {"schema": "research_runtime_status_v1", "run_id": manifest["run_id"],
                "status": status, "stage": current_stage, "attempt": str(attempt) if attempt else None,
                "controller": owner, "heartbeat_at": _now(), "stages": completed,
                "manifest_sha256": manifest_sha256, "stage_outcomes": outcomes,
                "stage_counts": {state: sum(row["status"] == state for row in outcomes.values())
                                 for state in ("pending", "running", "succeeded", "unavailable", "blocked", "failed", "stopped")},
                "execution_complete": status == "execution_complete", "research_complete": False,
                "execution_settled": status in {"execution_complete", "partial"},
                "promotion_authority": False, **extra})

    def record_failure(exc: BaseException) -> tuple[str, str]:
        blocker = str(exc) or type(exc).__name__
        status = "stopped" if blocker in {"safe_stop_requested", "stop_grace_exceeded"} or isinstance(exc, KeyboardInterrupt) else "failed"
        category = "engineering_failure"
        if blocker.startswith("pilot_not_ready:"):
            status, category = "blocked", "learning_or_evidence_not_ready"
        elif status == "stopped":
            category = "operator_stop"
        failure = {"run_id": manifest["run_id"], "status": status, "stage": current_stage,
                   "attempt": str(attempt) if attempt else None, "blocker": blocker,
                   "category": category, "exception_type": type(exc).__name__}
        if current_stage is not None:
            outcomes[current_stage] = {"status": status, "blocker": blocker}
        if attempt is not None:
            _atomic(attempt / "failure.json", {**failure, "finished_at": _now()})
        event_id = _event(root, failure)
        heartbeat(status, blocker=blocker, category=category, event_id=event_id, **latest_process)
        return status, blocker

    with _lock(root):
        # Recover visible failure, never inherit a stale "running" projection.
        previous = runtime_status(workspace, manifest["run_id"])
        if previous["status"] == "invalid_state":
            raise ValueError(previous["blocker"])
        if previous["status"] == "interrupted":
            _event(root, {"run_id": manifest["run_id"], "status": "interrupted", "attempt": previous.get("attempt"), "blocker": "controller_lost"})
        if resume and (root / "stop.json").exists():
            stopped = root / "stop_requests" / f"{time.time_ns()}.json"
            stopped.parent.mkdir(exist_ok=True)
            os.replace(root / "stop.json", stopped)
        _atomic(root / "manifests" / f"{manifest_sha256}.json", {
            "source_path": str(manifest_path.resolve()), "source_sha256": manifest_sha256, "manifest": manifest,
        })
        heartbeat("running")
        try:
            for stage in manifest["stages"]:
                current_stage, attempt = stage["id"], None
                latest_process = {}
                if (root / "stop.json").exists():
                    raise RuntimeError("safe_stop_requested")
                missing = [key for key in stage.get("depends_on", []) if key not in completed]
                if missing:
                    outcomes[current_stage] = {"status": "blocked", "blocker": "dependency_not_succeeded", "dependencies": missing}
                    heartbeat("running")
                    continue
                try:
                    outcomes[current_stage] = {"status": "running"}
                    heartbeat("running")
                    inputs = _inputs(workspace, manifest, stage)
                    if stage.get("unavailable"):
                        outcomes[current_stage] = {"status": "unavailable", **stage["unavailable"]}
                        heartbeat("running")
                        continue
                    dependencies = {key: completed[key] for key in stage.get("depends_on", [])}
                    binding = _digest({"run_id": manifest["run_id"], "stage": stage, "inputs": inputs,
                                       "dependencies": {k: v["sha256"] for k, v in dependencies.items()}, "platform": platform_binding})
                    # Resource-only amendments preserve unit identity, not stage
                    # acceptance. Source/code/commands/dependency bytes still bind.
                    work_binding = _digest({"run_id": manifest["run_id"], "inputs": inputs, "platform": platform_binding,
                        "stage": {k: v for k, v in stage.items() if k not in {"resources", "memory_pressure"}},
                        "dependencies": {k: {"work_binding_id": read_json(Path(v["receipt"])).get("work_binding_id", v["binding_id"]),
                                             "artifacts": read_json(Path(v["receipt"]))["artifacts"]} for k, v in dependencies.items()}})
                    reused = _reusable(root, stage, binding)
                    if reused:
                        completed[current_stage] = reused
                        outcomes[current_stage] = {"status": "succeeded", "reused": True}
                        heartbeat("running", reused_stage=True)
                        continue
                    attempt = root / "stages" / current_stage / "attempts" / f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
                    attempt.mkdir(parents=True)
                    _atomic(attempt / "binding.json", {"binding_id": binding, "work_binding_id": work_binding, "stage": stage,
                                                      "inputs": inputs, "dependencies": dependencies, "platform": platform_binding})
                    wait_start = time.monotonic()
                    resources = stage["resources"]
                    required = resources["max_memory_bytes"] + resources["available_floor_bytes"]
                    waited = False
                    while psutil.virtual_memory().available < required:
                        waited = True
                        heartbeat("waiting_resources", required_available_bytes=required,
                                  available_bytes=psutil.virtual_memory().available)
                        if (root / "stop.json").exists():
                            raise RuntimeError("safe_stop_requested")
                        if time.monotonic() - wait_start >= resources["wait_timeout_s"]:
                            raise RuntimeError("resource_wait_timeout")
                        time.sleep(poll_s)
                    resource_wait_s = time.monotonic() - wait_start
                    if waited:
                        _inputs(workspace, manifest, stage)
                    heartbeat("running", binding_id=binding)
                    child = _run_child(stage, workspace, attempt, binding, root, heartbeat, poll_s, work_binding)
                    latest_process = {k: child[k] for k in ("completed_units", "elapsed_s", "peak_rss_bytes", "stop_reason")}
                    _atomic(attempt / "process_result.json", child)
                    if child["blocker"]:
                        raise RuntimeError(child["blocker"])
                    _inputs(workspace, manifest, stage)
                    artifacts = {}
                    for name in stage["outputs"]:
                        path = artifact_path(attempt / "artifacts", name)
                        if not path.is_file() or not path.stat().st_size:
                            raise RuntimeError(f"missing_output:{name}")
                        artifacts[name] = file_digest(path)
                    audit = None
                    if stage["phase"] == "pilot":
                        audit = audit_readiness(attempt / "artifacts", stage["readiness_contract"], binding)
                        _atomic(attempt / "readiness_audit.json", audit)
                        if not audit["ready"]:
                            raise RuntimeError("pilot_not_ready:" + ";".join(audit["blockers"]))
                        artifacts.update(audit["artifacts"])
                    receipt = {"status": "succeeded", "run_id": manifest["run_id"], "binding_id": binding,
                               "work_binding_id": work_binding, "stage": current_stage,
                               "finished_at": _now(), "artifacts": artifacts, "process": child,
                               "readiness": audit, "resource_wait_s": resource_wait_s}
                    _atomic(attempt / "receipt.json", receipt)
                    ref = {"receipt": str((attempt / "receipt.json").relative_to(root)), "sha256": file_digest(attempt / "receipt.json")}
                    _atomic(root / "stages" / current_stage / "latest_success.json", ref)
                    completed[current_stage] = {**ref, "receipt": str(attempt / "receipt.json"), "binding_id": binding, "reused": False}
                    outcomes[current_stage] = {"status": "succeeded", "reused": False}
                except (Exception, KeyboardInterrupt) as exc:
                    status, _ = record_failure(exc)
                    if status == "stopped" or manifest.get("failure_policy", "fail_fast") == "fail_fast":
                        return runtime_status(workspace, manifest["run_id"])
            status = "execution_complete" if len(completed) == len(manifest["stages"]) else "partial"
            heartbeat(status, **latest_process)
            _event(root, {"run_id": manifest["run_id"], "status": status,
                          "stage_receipts": {k: v["sha256"] for k, v in completed.items()},
                          "stage_outcomes": {key: {k: v for k, v in row.items() if k != "reused"}
                                             for key, row in outcomes.items()}, "research_complete": False})
        except (Exception, KeyboardInterrupt) as exc:
            record_failure(exc)
    return runtime_status(workspace, manifest["run_id"])
