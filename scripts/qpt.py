"""Research runtime and event-delivery extraction of the source platform CLI."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def cmd_research_runtime(args: argparse.Namespace) -> int:
    from qplatform.research_runtime import (
        acknowledge_event, request_stop, run_manifest, runtime_status,
    )

    workspace = Path(args.workspace).resolve() if args.workspace else ROOT
    try:
        if args.subcmd == "run":
            out = run_manifest(workspace, Path(args.manifest).resolve(), resume=args.resume,
                               expected_run_id=args.run_id or None)
        elif args.subcmd == "status":
            out = runtime_status(workspace, args.run_id, compact=args.compact)
        elif args.subcmd == "stop":
            out = request_stop(workspace, args.run_id)
        elif args.subcmd == "ack":
            out = acknowledge_event(workspace, args.run_id, args.event_id, args.channel)
        else:
            raise ValueError("unknown research-runtime action")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid_request", "blocker": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return int(args.subcmd == "run" and out["status"] != "execution_complete")

def cmd_research_dispatch(args: argparse.Namespace) -> int:
    from qplatform.research_dispatch import (
        dispatch_root, dispatch_status, load_policy, process_event, start_dispatch, stop_dispatch,
    )
    from qplatform.research_observer import acknowledge_delivery, receive_delivery, resolve_cli
    from qplatform.research_runtime import _lock
    workspace = Path(args.workspace).resolve() if args.workspace else ROOT
    try:
        if args.subcmd == "start":
            out = start_dispatch(workspace, args.run_id, Path(args.manifest),
                                 Path(args.policy or ROOT / "configs/research_dispatch.json"),
                                 args.codex_exe or None, resume=args.resume)
        elif args.subcmd == "status":
            out = dispatch_status(workspace, args.run_id)
        elif args.subcmd == "stop":
            out = stop_dispatch(workspace, args.run_id)
        elif args.subcmd == "ack":
            out = acknowledge_delivery(workspace, args.run_id, args.event_id, args.thread_id)
        elif args.subcmd == "receive":
            out = receive_delivery(workspace, args.run_id, args.event_id, args.thread_id)
        else:
            root = dispatch_root(workspace, args.run_id)
            with _lock(root):
                request = {"policy": load_policy(Path(args.policy or ROOT / "configs/research_dispatch.json")),
                           "cli": str(resolve_cli(args.codex_exe or None))}
                out = process_event(workspace, args.run_id, request)
                if out.get("status") == "queued":
                    from qplatform.control_io import atomic_json
                    from qplatform.research_runtime import _now
                    atomic_json(root / "current.json", {**out, "run_id": args.run_id,
                                                        "finished_at": _now()})
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "blocker": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 2 if args.subcmd == "status" and out.get("status") == "at_risk" else 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("research-runtime", help="bounded research stages, pilot gates and failure receipts")
    runtime_sub = runtime.add_subparsers(dest="subcmd", required=True)
    for action in ("run", "status", "stop", "ack"):
        command = runtime_sub.add_parser(action)
        command.add_argument("--workspace", default="", help="workspace containing the frozen run, not a data refresh")
        if action == "run":
            command.add_argument("--manifest", required=True)
            command.add_argument("--run-id", required=True, help="reject a manifest for a different research run")
            command.add_argument("--resume", action="store_true", help="explicitly clear a prior safe-stop request")
        else:
            command.add_argument("--run-id", required=True)
        if action == "ack":
            command.add_argument("--event-id", required=True)
            command.add_argument("--channel", required=True)
        if action == "status":
            command.add_argument("--compact", action="store_true", help="current process facts only, no historical round-local fields")
        command.set_defaults(func=cmd_research_runtime)

    dispatch = sub.add_parser("research-dispatch", help="detached runtime, bounded observer and event handoff")
    dispatch_sub = dispatch.add_subparsers(dest="subcmd", required=True)
    for action in ("start", "status", "stop", "handoff", "ack", "receive"):
        command = dispatch_sub.add_parser(action)
        command.add_argument("--workspace", default="")
        command.add_argument("--run-id", required=True)
        if action in {"start", "handoff"}:
            command.add_argument("--policy", required=True, help="explicit routing policy; may invoke a configured CLI")
            command.add_argument("--codex-exe", default="")
        if action == "start":
            command.add_argument("--manifest", required=True)
            command.add_argument("--resume", action="store_true")
        if action in {"ack", "receive"}:
            command.add_argument("--event-id", required=True)
            command.add_argument("--thread-id", required=True)
        command.set_defaults(func=cmd_research_dispatch)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
