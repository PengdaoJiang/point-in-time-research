from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dependence_statistics import dependence_aware_mean_inference
from .time_series_diagnostics import build_serial_dependence_diagnostics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_daily_returns(
    frame: pd.DataFrame,
    *,
    date_column: str,
    return_column: str,
) -> pd.DataFrame:
    missing = [column for column in (date_column, return_column) if column not in frame]
    if missing:
        raise ValueError(f"daily return input missing columns: {missing}")

    out = frame.loc[:, [date_column, return_column]].copy()
    out.columns = ["date", "net_return"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("daily return input contains invalid dates")
    if out["net_return"].isna().any() or not np.isfinite(out["net_return"]).all():
        raise ValueError("daily return input contains non-finite returns")
    if out["date"].duplicated().any():
        duplicates = (
            out.loc[out["date"].duplicated(keep=False), "date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        raise ValueError(f"daily return input contains duplicate dates: {duplicates[:5]}")
    if (out["net_return"] <= -1.0).any():
        raise ValueError("daily return input contains a return at or below -100%")
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def path_metrics(returns: Iterable[float], *, periods_per_year: int = 252) -> dict[str, Any]:
    values = np.asarray(list(returns), dtype=float)
    if values.size == 0:
        return {
            "observations": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "worst_day": None,
            "win_rate": None,
        }

    curve = np.cumprod(1.0 + values)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], curve)))
    drawdown = np.concatenate(([1.0], curve)) / running_peak - 1.0
    total_return = float(curve[-1] - 1.0)
    annualized_return = float(
        (1.0 + total_return) ** (float(periods_per_year) / float(values.size)) - 1.0
    )
    daily_std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    annualized_volatility = daily_std * math.sqrt(float(periods_per_year))
    sharpe = (
        float(values.mean() / daily_std * math.sqrt(float(periods_per_year)))
        if daily_std > 0.0
        else None
    )
    return {
        "observations": int(values.size),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(annualized_volatility),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "worst_day": float(values.min()),
        "win_rate": float((values > 0.0).mean()),
    }


def remove_best_days_total_return(
    returns: Iterable[float],
    remove_count: int,
) -> float | None:
    values = np.asarray(list(returns), dtype=float)
    if values.size == 0 or int(remove_count) >= values.size:
        return None
    if int(remove_count) > 0:
        remove_indices = np.argsort(values)[-int(remove_count) :]
        values = np.delete(values, remove_indices)
    return float(np.prod(1.0 + values) - 1.0)


def overlapping_window_table(
    daily: pd.DataFrame,
    *,
    horizons: Iterable[int],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        if horizon <= 0:
            raise ValueError(f"window horizon must be positive: {horizon}")
        if daily.shape[0] < horizon:
            continue
        for end_index in range(horizon - 1, daily.shape[0]):
            start_index = end_index - horizon + 1
            window = daily.iloc[start_index : end_index + 1]
            rows.append(
                {
                    "horizon_observations": horizon,
                    "start_date": window["date"].iloc[0].date().isoformat(),
                    "end_date": window["date"].iloc[-1].date().isoformat(),
                    **path_metrics(
                        window["net_return"].to_numpy(dtype=float),
                        periods_per_year=periods_per_year,
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_windows(windows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_observations",
        "window_count",
        "positive_window_fraction",
        "mean_total_return",
        "median_total_return",
        "minimum_total_return",
        "p10_total_return",
        "p90_total_return",
        "maximum_total_return",
        "median_sharpe",
        "worst_window_drawdown",
        "latest_window_end_date",
        "latest_window_total_return",
    ]
    if windows.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for horizon, group in windows.groupby("horizon_observations", sort=True):
        ordered = group.sort_values("end_date", kind="mergesort")
        total_returns = pd.to_numeric(ordered["total_return"], errors="coerce")
        sharpes = pd.to_numeric(ordered["sharpe"], errors="coerce")
        drawdowns = pd.to_numeric(ordered["max_drawdown"], errors="coerce")
        rows.append(
            {
                "horizon_observations": int(horizon),
                "window_count": int(ordered.shape[0]),
                "positive_window_fraction": float((total_returns > 0.0).mean()),
                "mean_total_return": float(total_returns.mean()),
                "median_total_return": float(total_returns.median()),
                "minimum_total_return": float(total_returns.min()),
                "p10_total_return": float(total_returns.quantile(0.10)),
                "p90_total_return": float(total_returns.quantile(0.90)),
                "maximum_total_return": float(total_returns.max()),
                "median_sharpe": (
                    float(sharpes.median()) if sharpes.notna().any() else None
                ),
                "worst_window_drawdown": float(drawdowns.min()),
                "latest_window_end_date": str(ordered["end_date"].iloc[-1]),
                "latest_window_total_return": float(total_returns.iloc[-1]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def calendar_period_table(
    daily: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    tagged = daily.copy()
    tagged["calendar_year"] = tagged["date"].dt.year
    tagged["calendar_quarter"] = tagged["date"].dt.to_period("Q").astype(str)
    rows: list[dict[str, Any]] = []
    for period_kind, column in (
        ("calendar_year", "calendar_year"),
        ("calendar_quarter", "calendar_quarter"),
    ):
        for period, group in tagged.groupby(column, sort=True):
            rows.append(
                {
                    "period_kind": period_kind,
                    "period": str(period),
                    "start_date": group["date"].iloc[0].date().isoformat(),
                    "end_date": group["date"].iloc[-1].date().isoformat(),
                    **path_metrics(
                        group["net_return"].to_numpy(dtype=float),
                        periods_per_year=periods_per_year,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_sequential_validation(
    daily: pd.DataFrame,
    *,
    candidate_id: str,
    horizons: Iterable[int],
    causal_signal_path_verified: bool,
    selection_posture: str,
    periods_per_year: int = 252,
    overlap_horizon: int = 1,
    bootstrap_resamples: int = 1000,
    diagnostic_max_lag: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_horizons = sorted({int(value) for value in horizons})
    windows = overlapping_window_table(
        daily,
        horizons=normalized_horizons,
        periods_per_year=periods_per_year,
    )
    window_summary = summarize_windows(windows)
    calendar_periods = calendar_period_table(
        daily,
        periods_per_year=periods_per_year,
    )
    values = daily["net_return"].to_numpy(dtype=float)
    full_metrics = path_metrics(values, periods_per_year=periods_per_year)
    full_metrics.update(
        {
            "remove_best_1_day_total_return": remove_best_days_total_return(values, 1),
            "remove_best_5_days_total_return": remove_best_days_total_return(values, 5),
            "remove_best_10_days_total_return": remove_best_days_total_return(values, 10),
        }
    )
    full_metrics["dependence_aware_mean_inference"] = (
        dependence_aware_mean_inference(
            values,
            overlap_horizon=overlap_horizon,
            bootstrap_resamples=bootstrap_resamples,
        )
    )
    full_metrics["serial_dependence_diagnostics"] = (
        build_serial_dependence_diagnostics(
            values,
            max_lag=diagnostic_max_lag,
        )
    )
    result = {
        "schema_version": 3,
        "candidate_id": str(candidate_id),
        "status": (
            "sequential_multi_origin_stability_computed"
            if causal_signal_path_verified
            else "diagnostic_only_causal_signal_path_not_verified"
        ),
        "date_min": daily["date"].iloc[0].date().isoformat(),
        "date_max": daily["date"].iloc[-1].date().isoformat(),
        "full_period_metrics": full_metrics,
        "temporal_evaluation": {
            "method": "all_history_overlapping_multi_origin_windows",
            "horizons_observations": normalized_horizons,
            "single_chronological_split_is_sole_evidence": False,
            "fixed_future_observation_signal_gate": False,
            "future_observations_are_monitoring_and_stop_evidence_only": True,
            "causal_signal_path_verified": bool(causal_signal_path_verified),
            "selection_posture": str(selection_posture),
            "candidate_selection_bias_removed": False,
            "daily_iid_inference_eligible": False,
            "dependence_inference": (
                "date-level Newey-West HAC plus circular moving-block bootstrap"
            ),
            "serial_dependence_diagnostic_required": True,
            "zero_return_acf_proves_unpredictable": False,
            "full_sample_acf_pacf_order_selection_is_confirmation_evidence": False,
            "overlap_horizon_observations": int(overlap_horizon),
            "interpretation": (
                "Current-history temporal stability evidence. It does not make a "
                "post-selected candidate pristine and creates no signal or order authority."
            ),
        },
        "window_summary": window_summary.to_dict(orient="records"),
        "calendar_positive_fraction": {
            period_kind: float(
                (
                    pd.to_numeric(
                        calendar_periods.loc[
                            calendar_periods["period_kind"].eq(period_kind),
                            "total_return",
                        ],
                        errors="coerce",
                    )
                    > 0.0
                ).mean()
            )
            for period_kind in ("calendar_year", "calendar_quarter")
        },
        "paper_or_live_authority": False,
        "orders_generated": 0,
        "canonical_writes": 0,
    }
    return result, windows, window_summary, calendar_periods


def write_sequential_validation(
    *,
    source_csv: Path,
    output_dir: Path,
    candidate_id: str,
    date_column: str,
    return_column: str,
    horizons: Iterable[int],
    causal_signal_path_verified: bool,
    selection_posture: str,
    periods_per_year: int = 252,
    overlap_horizon: int = 1,
    bootstrap_resamples: int = 1000,
    diagnostic_max_lag: int = 20,
) -> dict[str, Any]:
    source_csv = source_csv.resolve()
    output_dir = output_dir.resolve()
    daily = prepare_daily_returns(
        pd.read_csv(source_csv),
        date_column=date_column,
        return_column=return_column,
    )
    result, windows, window_summary, calendar_periods = build_sequential_validation(
        daily,
        candidate_id=candidate_id,
        horizons=horizons,
        causal_signal_path_verified=causal_signal_path_verified,
        selection_posture=selection_posture,
        periods_per_year=periods_per_year,
        overlap_horizon=overlap_horizon,
        bootstrap_resamples=bootstrap_resamples,
        diagnostic_max_lag=diagnostic_max_lag,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "overlapping_windows.csv"
    summary_path = output_dir / "window_summary.csv"
    calendar_path = output_dir / "calendar_periods.csv"
    manifest_path = output_dir / "sequential_validation.json"
    markdown_path = output_dir / "summary.md"
    windows.to_csv(windows_path, index=False, encoding="utf-8")
    window_summary.to_csv(summary_path, index=False, encoding="utf-8")
    calendar_periods.to_csv(calendar_path, index=False, encoding="utf-8")

    result["input"] = {
        "source_csv": str(source_csv),
        "source_sha256": sha256_file(source_csv),
        "date_column": date_column,
        "return_column": return_column,
    }
    result["paths"] = {
        "overlapping_windows": str(windows_path),
        "window_summary": str(summary_path),
        "calendar_periods": str(calendar_path),
        "summary": str(markdown_path),
        "manifest": str(manifest_path),
    }
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    full = result["full_period_metrics"]
    dependence = full["dependence_aware_mean_inference"]
    serial = full["serial_dependence_diagnostics"]
    return_memory = serial["series"]["returns"]
    hac = dependence["hac"]
    block = dependence["block_bootstrap"]
    lines = [
        "# Sequential Multi-Origin Stability",
        "",
        f"- candidate_id: `{candidate_id}`",
        f"- status: `{result['status']}`",
        f"- date_range: `{result['date_min']}` to `{result['date_max']}`",
        f"- observations: `{full['observations']}`",
        f"- total_return: `{full['total_return']:.6f}`",
        f"- max_drawdown: `{full['max_drawdown']:.6f}`",
        f"- remove_best_5_days_total_return: `{full['remove_best_5_days_total_return']}`",
        f"- effective_observations: `{hac['effective_observations']}`",
        f"- hac_lag: `{hac['hac_lag']}`",
        f"- hac_t_stat: `{hac['hac_t_stat']}`",
        f"- block_bootstrap_status: `{block['status']}`",
        f"- block_length: `{block['block_length']}`",
        (
            "- block_bootstrap_95pct_ci: "
            f"`[{block['confidence_interval_low']}, "
            f"{block['confidence_interval_high']}]`"
        ),
        f"- serial_diagnostic_max_lag: `{serial['predeclared_max_lag']}`",
        f"- strongest_absolute_return_acf_lag: `{return_memory['strongest_absolute_acf_lag']}`",
        f"- strongest_absolute_return_acf: `{return_memory['strongest_absolute_acf']}`",
        f"- return_ljung_box_p_value: `{return_memory['ljung_box']['p_value']}`",
        f"- volatility_memory_detected: `{serial['summary']['volatility_memory_detected']}`",
        f"- selection_posture: `{selection_posture}`",
        "",
        "This report uses every currently observable date through overlapping",
        "multi-origin windows. No future observation count is an admission gate.",
        "ACF/PACF are linear-memory diagnostics, not general predictability",
        "tests. Any lag/order found on the full history must be frozen before",
        "causal confirmation.",
        "It creates no paper, live, or order authority.",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return result
