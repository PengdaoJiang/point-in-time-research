"""Runtime-only extraction of the source platform CLI."""
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

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
