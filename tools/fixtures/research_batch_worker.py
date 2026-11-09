"""Deterministic infrastructure fixture, not a strategy or scientific pilot."""
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--value", type=int, required=True)
parser.add_argument("--delay", type=float, default=.1)
parser.add_argument("--fail-unless")
args = parser.parse_args()
if args.fail_unless and not Path(args.fail_unless).exists():
    raise SystemExit(3)
time.sleep(args.delay)
out = Path(os.environ["QPT_OUTPUT_DIR"])
(out / "result.json").write_text(json.dumps({"value": args.value**2, "threads": os.environ["OMP_NUM_THREADS"]}))
