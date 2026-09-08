"""Replay a synthetic data-quality/temporal-evaluation case through real code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qplatform.point_in_time_universe import resolve_asof_membership
from qplatform.research_guard import audit_research_feature_columns
from qplatform.sequential_validation import write_sequential_validation


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    snapshots = [
        {"date": "2026-01-02", "symbols": ["000001", "000002"]},
        {"date": "2026-01-06", "symbols": ["000002", "000003"]},
    ]
    membership, coverage = resolve_asof_membership(
        snapshots, ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-12"], max_age_days=4
    )
    membership.to_csv(output / "membership.csv", index=False)
    coverage.to_csv(output / "coverage.csv", index=False)
    dates = pd.bdate_range("2025-01-02", periods=120)
    rng = np.random.default_rng(73)
    returns = rng.normal(0, 0.007, len(dates))
    returns[1:] += 0.35 * returns[:-1]
    pd.DataFrame({"date": dates, "net_return": returns}).to_csv(output / "synthetic_returns.csv", index=False)
    result = write_sequential_validation(
        source_csv=output / "synthetic_returns.csv", output_dir=output / "validation",
        candidate_id="synthetic_pipeline_case", date_column="date", return_column="net_return",
        horizons=[10, 20, 40], causal_signal_path_verified=False,
        selection_posture="synthetic_fixture_not_a_strategy", overlap_horizon=3, bootstrap_resamples=200,
    )
    evidence = {
        "data_kind": "synthetic", "purpose": "exercise source-derived pipeline, not demonstrate alpha",
        "membership_rejections": coverage.loc[~coverage["covered"], "reason"].tolist(),
        "feature_audit": audit_research_feature_columns(["ret_5d", "amount", "future_return", "cash_available_cny"]),
        "validation_status": result["status"], "report": "validation/summary.md",
    }
    (output / "case.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo-output"))
    print(json.dumps(run(parser.parse_args().out), ensure_ascii=False, indent=2))
