"""Engineering acceptance of a real research pilot, not strategy promotion."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def artifact_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or path == directory.resolve():
        raise ValueError("artifact escapes its attempt directory")
    return path


def _artifact(directory: Path, reference: dict[str, Any]) -> Path:
    path = artifact_path(directory, reference["path"])
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing/empty evidence: {path.name}")
    if file_digest(path) != reference["sha256"]:
        raise ValueError(f"evidence hash mismatch: {path.name}")
    return path


def _clock(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("evidence clocks require explicit timezone")
    return result


def compare_scenario_maturity(
    expected: dict[str, dict[str, int]], observed: dict[str, dict[str, int]]
) -> list[str]:
    """Compare each scenario to its own frozen reference, never a global max."""
    blockers = []
    if expected.keys() != observed.keys():
        blockers.append("scenario_coverage_mismatch")
    for scenario, counts in expected.items():
        if counts != observed.get(scenario):
            blockers.append(f"{scenario}:maturity_reference_mismatch")
    return blockers


def audit_label_support(
    rows: list[dict[str, Any]], *, cutoff: str, required_roles: list[str],
) -> dict[str, Any]:
    """Cheap pre-fit capability check; an available WAIT is not an action label.

    Callers normalize their own domain action names into frozen roles. This
    examines label availability only, and does not claim a model was fitted.
    """
    clock = _clock(cutoff)
    counts: dict[str, int] = {}
    unresolved = 0
    not_mature = 0
    blockers = []
    for row in rows:
        status = row["status"]
        value = row["value"]
        if status not in {"RESOLVED_MARKET", "RESOLVED_CASH"}:
            unresolved += 1
            if value is not None and str(value).strip() and math.isfinite(float(value)):
                blockers.append("unresolved_label_has_numeric_return")
            continue
        if value is None or not math.isfinite(float(value)):
            blockers.append("resolved_label_missing_or_nonfinite")
            continue
        if status == "RESOLVED_CASH" and float(value) != 0:
            blockers.append("resolved_cash_not_zero")
        if _clock(row["available_at"]) >= clock:
            not_mature += 1
            continue
        role = row["training_role"]
        counts[role] = counts.get(role, 0) + 1
    for role in required_roles:
        if not counts.get(role):
            blockers.append(f"no_observable_label_support_for_role:{role}")
    if not required_roles:
        blockers.append("required_roles_not_declared")
    return {"support_available": not blockers, "observable_labels_by_role": counts,
            "unresolved": unresolved, "not_yet_mature": not_mature,
            "blockers": sorted(set(blockers)), "actual_fit_verified": False}


def audit_readiness(
    directory: Path, contract: dict[str, Any], binding_id: str,
    evidence_name: str = "evidence.json",
) -> dict[str, Any]:
    """Read hash-bound pilot labels, actual fit traces and account observations.

    This detects absent learning and contradictory receipts. It cannot certify
    the adapter's financial semantics; its independent causal audit is still
    required. Negative returns are never an engineering blocker.
    """
    blockers: list[str] = []
    reports: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    try:
        evidence_path = artifact_path(directory, evidence_name)
        evidence = read_json(evidence_path)
        artifacts[evidence_name] = file_digest(evidence_path)
        if evidence["binding_id"] != binding_id:
            blockers.append("pilot_binding_mismatch")
        if evidence["schema"] != "research_readiness_v1":
            blockers.append("unsupported_evidence_schema")
        required = contract["scenarios"]
        observed = evidence["scenarios"]
        if not required:
            raise ValueError("empty scenario contract")
        blockers.extend(compare_scenario_maturity(
            {key: value["maturity_by_window"] for key, value in required.items()},
            {key: value["maturity_by_window"] for key, value in observed.items()},
        ))
        for scenario, spec in required.items():
            if scenario not in observed:
                continue
            row = observed[scenario]
            local: list[str] = []
            counts = {"resolved_market": 0, "resolved_cash": 0, "unresolved": 0}

            def checked(ref: dict[str, Any]) -> Path:
                path = _artifact(directory, ref)
                artifacts[ref["path"]] = ref["sha256"]
                return path

            with checked(row["labels"]).open(encoding="utf-8-sig", newline="") as stream:
                labels = {}
                for label in csv.DictReader(stream):
                    key = label["label_id"]
                    if not key or key in labels:
                        raise ValueError(f"{scenario}:duplicate/empty label_id")
                    status = label["status"]
                    value = label["value"].strip()
                    if status in {"RESOLVED_MARKET", "RESOLVED_CASH"}:
                        if not value or not math.isfinite(float(value)):
                            local.append("resolved_label_missing_or_nonfinite")
                        elif status == "RESOLVED_CASH" and float(value) != 0:
                            local.append("resolved_cash_not_zero")
                        _clock(label["available_at"])
                        counts["resolved_market" if status == "RESOLVED_MARKET" else "resolved_cash"] += 1
                    elif status in {"UNRESOLVED", "RIGHT_CENSORED", "UNKNOWN_SOURCE", "UNKNOWN_EXECUTION"}:
                        counts["unresolved"] += 1
                        if value and math.isfinite(float(value)):
                            local.append("unresolved_label_has_numeric_return")
                    else:
                        local.append("unknown_label_status")
                    labels[key] = label

            fits = []
            if row.get("fits"):
                with checked(row["fits"]).open(encoding="utf-8") as stream:
                    fits = [json.loads(line) for line in stream if line.strip()]
            train_rows = 0
            market_train_rows = 0
            training_roles: dict[str, int] = {}
            fit_ids = set()
            for fit in fits:
                if fit["fit_id"] in fit_ids:
                    local.append("duplicate_fit_id")
                fit_ids.add(fit["fit_id"])
                decision = _clock(fit["decision_at"])
                ids = fit["training_label_ids"]
                if not ids or len(ids) != len(set(ids)) or fit["fitted_rows"] != len(ids):
                    local.append("invalid_actual_fit_rows")
                checked(fit["model"])
                for label_id in ids:
                    label = labels.get(label_id)
                    if label is None or label["status"] not in {"RESOLVED_MARKET", "RESOLVED_CASH"}:
                        local.append("fit_consumed_unknown_label")
                        continue
                    if _clock(label["available_at"]) >= decision:
                        local.append("fit_consumed_unobservable_label")
                    train_rows += 1
                    market_train_rows += int(label["status"] == "RESOLVED_MARKET")
                    role = label["training_role"]
                    if not role:
                        local.append("missing_training_role")
                    training_roles[role] = training_roles.get(role, 0) + 1
            mode = spec["mode"]
            if mode not in {"fit", "replay"}:
                raise ValueError(f"invalid readiness mode: {mode}")
            if mode == "fit" and (not fits or not train_rows or not market_train_rows):
                local.append("no_effective_learning")
            if mode == "fit":
                if not spec.get("required_training_roles"):
                    local.append("missing_required_training_roles")
                for role in spec.get("required_training_roles", []):
                    if not training_roles.get(role):
                        local.append(f"no_training_support_for_role:{role}")

            account_rows = 0
            with checked(row["account"]).open(encoding="utf-8-sig", newline="") as stream:
                previous = None
                for account in csv.DictReader(stream):
                    clock = _clock(account["observed_at"])
                    if previous is not None and clock <= previous:
                        local.append("account_clock_not_increasing")
                    previous = clock
                    cash, positions, equity = (float(account[key]) for key in ("cash", "position_value", "equity"))
                    if not all(math.isfinite(x) for x in (cash, positions, equity)):
                        local.append("nonfinite_account_value")
                    elif abs(cash + positions - equity) > max(.01, abs(equity) * 1e-9):
                        local.append("account_balance_mismatch")
                    account_rows += 1
            if not account_rows:
                local.append("empty_account_evidence")
            reports[scenario] = {
                **counts, "fit_count": len(fits), "actual_training_rows": train_rows,
                "market_training_rows": market_train_rows, "account_rows": account_rows,
                "training_rows_by_role": training_roles,
                "blockers": sorted(set(local)),
            }
            blockers.extend(f"{scenario}:{item}" for item in sorted(set(local)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        blockers.append(f"invalid_or_missing_evidence:{type(exc).__name__}:{exc}")
    return {
        "schema": "research_readiness_audit_v1", "ready": not blockers,
        "blockers": sorted(set(blockers)), "scenarios": reports,
        "artifacts": artifacts, "binding_id": binding_id,
        "strategy_profitability_assessed": False, "promotion_authority": False,
    }
