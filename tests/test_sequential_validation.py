from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qplatform.sequential_validation import (
    build_sequential_validation,
    prepare_daily_returns,
)


def test_positive_path_uses_all_overlapping_origins() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "net_return": [0.01] * 10,
        }
    )
    result, windows, summary, calendar = build_sequential_validation(
        daily,
        candidate_id="synthetic_positive",
        horizons=[3, 5],
        causal_signal_path_verified=True,
        selection_posture="test_only",
    )
    assert windows.loc[windows["horizon_observations"].eq(3)].shape[0] == 8
    assert windows.loc[windows["horizon_observations"].eq(5)].shape[0] == 6
    assert summary["positive_window_fraction"].eq(1.0).all()
    assert not calendar.empty
    temporal = result["temporal_evaluation"]
    assert temporal["single_chronological_split_is_sole_evidence"] is False
    assert temporal["fixed_future_observation_signal_gate"] is False
    assert temporal["daily_iid_inference_eligible"] is False
    dependence = result["full_period_metrics"][
        "dependence_aware_mean_inference"
    ]
    assert dependence["method"].startswith("date_level_newey_west")
    assert dependence["hac"]["nominal_observations"] == 10
    assert dependence["block_bootstrap"]["block_length"] >= 1
    serial = result["full_period_metrics"]["serial_dependence_diagnostics"]
    assert serial["method"].startswith("acf_pacf_ljung_box")
    assert serial["interpretation"]["zero_return_acf_proves_unpredictable"] is False
    assert result["schema_version"] == 3
    assert result["paper_or_live_authority"] is False


def test_overlap_horizon_drives_hac_and_block_floors() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "net_return": [0.001, -0.0005] * 20,
        }
    )
    result, _, _, _ = build_sequential_validation(
        daily,
        candidate_id="overlap_floor",
        horizons=[5],
        causal_signal_path_verified=True,
        selection_posture="test_only",
        overlap_horizon=8,
        bootstrap_resamples=100,
    )
    inference = result["full_period_metrics"][
        "dependence_aware_mean_inference"
    ]
    assert inference["hac"]["hac_lag"] >= 7
    assert inference["block_bootstrap"]["block_length"] >= 8


def test_duplicate_dates_fail_closed() -> None:
    frame = pd.DataFrame(
        {
            "when": ["2026-01-01", "2026-01-01"],
            "ret": [0.01, 0.02],
        }
    )
    try:
        prepare_daily_returns(
            frame,
            date_column="when",
            return_column="ret",
        )
    except ValueError as exc:
        assert "duplicate dates" in str(exc)
    else:
        raise AssertionError("duplicate dates must fail closed")


def test_unverified_causal_path_is_diagnostic_only() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "net_return": [0.0] * 5,
        }
    )
    result, _, _, _ = build_sequential_validation(
        daily,
        candidate_id="unverified",
        horizons=[3],
        causal_signal_path_verified=False,
        selection_posture="unknown",
    )
    assert result["status"] == "diagnostic_only_causal_signal_path_not_verified"


if __name__ == "__main__":
    test_positive_path_uses_all_overlapping_origins()
    test_overlap_horizon_drives_hac_and_block_floors()
    test_duplicate_dates_fail_closed()
    test_unverified_causal_path_is_diagnostic_only()
    print("PASS: sequential validation contract")
