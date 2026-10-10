"""Deterministic real child process used only by runtime fault-injection tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    mode = sys.argv[1]
    output = Path(os.environ["QPT_OUTPUT_DIR"])
    if mode in {"units", "units_crash_once", "units_controller_crash_once"}:
        from qplatform.research_checkpoints import CheckpointStore
        from qplatform.research_runtime import _atomic
        store = CheckpointStore.from_environment()
        results, reused = [], 0
        for index in range(3):
            unit_id = f"account-{index}"
            saved = store.load(unit_id)
            if saved is None:
                directory = output / unit_id
                directory.mkdir()
                row = {"cash": 150000 - (index + 1) * 100, "position_value": 0,
                       "equity": 150000 - (index + 1) * 100}
                (directory / "account.json").write_text(json.dumps(row))
                with Path("computed_units.jsonl").open("a") as log:
                    log.write(json.dumps({"unit_id": unit_id, "binding": store.binding}) + "\n")
                saved = store.seal(unit_id, directory, ["account.json"])
            else:
                reused += 1
            results.append(json.loads((saved["directory"] / "account.json").read_text()))
            _atomic(output / "progress.json", {"completed_units": index + 1})
            if mode == "units_crash_once" and index == 0 and not Path("failure_injected").exists():
                Path("failure_injected").touch()
                os._exit(17)
            if mode == "units_controller_crash_once" and index == 0 and not Path("failure_injected").exists():
                Path("failure_injected").touch()
                time.sleep(60)
        (output / "result.json").write_text(json.dumps({"accounts": results, "reused_units": reused}))
        return
    if mode == "fail":
        raise RuntimeError("injected stage failure")
    if mode in {"hang", "cooperative", "spawn"}:
        if mode == "spawn":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            (output / "grandchild.json").write_text(json.dumps({"pid": child.pid}))
        while True:
            if mode == "cooperative" and Path(os.environ["QPT_STOP_FILE"]).exists():
                (output / "checkpoint.json").write_text('{"complete_boundary":true}')
                return
            time.sleep(.05)
    if mode == "scale":
        (output / "result.json").write_text('{"computed":true}')
        return
    if mode == "progress":
        from qplatform.research_runtime import _atomic
        _atomic(output / "progress.json", {"completed_units": 1})
        time.sleep(.6)
        (output / "result.json").write_text('{"computed":true}')
        return
    unknown = mode == "empty_fit"
    future = mode == "future"
    model = output / "model.json"
    values = [-.02, -.01]
    model.write_text(json.dumps({"mean": sum(values) / len(values)}))

    def reference(name: str) -> dict:
        return {"path": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}

    scenarios = {}
    for scenario, maturity in {"L2": {"5": 4, "10": 9, "20": 19}, "L5": {"5": 4, "10": 9, "20": 16}}.items():
        label_name, fit_name, account_name = (f"{scenario}_{kind}.{ext}" for kind, ext in (("labels", "csv"), ("fits", "jsonl"), ("account", "csv")))
        with (output / label_name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["label_id", "status", "value", "available_at", "training_role"])
            writer.writeheader()
            for i, value in enumerate(values):
                writer.writerow({"label_id": str(i), "status": "UNRESOLVED" if unknown else "RESOLVED_MARKET",
                                 "value": "" if unknown else value, "available_at": "2026-09-01T10:00:00+08:00",
                                 "training_role": "wait" if mode == "wait_only" else "action"})
        fit = {"fit_id": "fit0", "decision_at": "2026-09-01T09:59:00+08:00" if future else "2026-09-02T09:30:00+08:00",
               "training_label_ids": ["0", "1"], "fitted_rows": 2, "model": reference("model.json")}
        (output / fit_name).write_text(json.dumps({"placeholder": "no actual fit"}) if unknown else json.dumps(fit))
        with (output / account_name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["observed_at", "cash", "position_value", "equity"])
            writer.writeheader()
            writer.writerow({"observed_at": "2026-09-02T15:00:00+08:00", "cash": 140000, "position_value": 0, "equity": 140000})
        scenarios[scenario] = {"maturity_by_window": maturity, "labels": reference(label_name),
                               "fits": None if unknown or mode == "rule_replay" else reference(fit_name),
                               "account": reference(account_name)}
    evidence = {"schema": "research_readiness_v1", "binding_id": os.environ["QPT_BINDING_ID"], "scenarios": scenarios}
    (output / "evidence.json").write_text(json.dumps(evidence))


if __name__ == "__main__":
    main()
