from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


_MINIMUM_BOOTSTRAP_EFFECTIVE_BLOCKS = 4.0
_MINIMUM_BOOTSTRAP_UNIQUE_STATISTICS = 20


def finite_values(values: Iterable[float]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    out = numeric.to_numpy(dtype=float)
    return out[np.isfinite(out)]


def recommended_hac_lag(
    observations: int,
    *,
    overlap_horizon: int = 1,
    minimum_lag: int = 0,
) -> int:
    n = max(int(observations), 0)
    if n <= 1:
        return 0
    overlap_floor = max(int(overlap_horizon) - 1, 0)
    rule_of_thumb = int(math.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0)))
    return min(n - 1, max(overlap_floor, int(minimum_lag), rule_of_thumb))


def newey_west_mean_inference(
    values: Iterable[float],
    *,
    overlap_horizon: int = 1,
    minimum_lag: int = 0,
) -> dict[str, Any]:
    array = finite_values(values)
    n = int(array.size)
    lag = recommended_hac_lag(
        n,
        overlap_horizon=overlap_horizon,
        minimum_lag=minimum_lag,
    )
    empty = {
        "nominal_observations": n,
        "effective_observations": None,
        "mean": None,
        "iid_standard_error_diagnostic": None,
        "iid_t_stat_diagnostic": None,
        "hac_lag": lag,
        "hac_long_run_variance": None,
        "hac_standard_error": None,
        "hac_t_stat": None,
        "hac_two_sided_p_value": None,
        "hac_positive_one_sided_p_value": None,
        "hac_negative_one_sided_p_value": None,
        "lag_autocorrelations": [],
    }
    if n == 0:
        return empty

    mean = float(array.mean())
    empty["mean"] = mean
    if n == 1:
        return empty

    centered = array - mean
    sample_variance = float(array.var(ddof=1))
    iid_standard_error = math.sqrt(sample_variance / float(n))
    iid_t_stat = (
        mean / iid_standard_error if iid_standard_error > 1e-15 else None
    )

    gamma_zero = float(np.dot(centered, centered) / float(n))
    long_run_variance = gamma_zero
    autocorrelations: list[float] = []
    for offset in range(1, lag + 1):
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / float(n)
        )
        weight = 1.0 - float(offset) / float(lag + 1)
        long_run_variance += 2.0 * weight * covariance
        autocorrelations.append(
            covariance / gamma_zero if gamma_zero > 1e-15 else float("nan")
        )
    long_run_variance = max(float(long_run_variance), 0.0)
    hac_standard_error = math.sqrt(long_run_variance / float(n))
    hac_t_stat = mean / hac_standard_error if hac_standard_error > 1e-15 else None
    effective_observations = None
    if long_run_variance > 1e-15 and sample_variance > 0.0:
        effective_observations = float(
            np.clip(float(n) * sample_variance / long_run_variance, 1.0, float(n))
        )

    if hac_t_stat is None:
        two_sided = positive = negative = None
    else:
        two_sided = float(math.erfc(abs(hac_t_stat) / math.sqrt(2.0)))
        positive = float(0.5 * math.erfc(hac_t_stat / math.sqrt(2.0)))
        negative = float(0.5 * math.erfc(-hac_t_stat / math.sqrt(2.0)))

    return {
        "nominal_observations": n,
        "effective_observations": effective_observations,
        "mean": mean,
        "iid_standard_error_diagnostic": iid_standard_error,
        "iid_t_stat_diagnostic": iid_t_stat,
        "hac_lag": lag,
        "hac_long_run_variance": long_run_variance,
        "hac_standard_error": hac_standard_error,
        "hac_t_stat": hac_t_stat,
        "hac_two_sided_p_value": two_sided,
        "hac_positive_one_sided_p_value": positive,
        "hac_negative_one_sided_p_value": negative,
        "lag_autocorrelations": autocorrelations,
    }


def audit_block_bootstrap_design(
    *,
    observations: int,
    block_length: int,
    minimum_effective_blocks: float = _MINIMUM_BOOTSTRAP_EFFECTIVE_BLOCKS,
) -> dict[str, Any]:
    n = max(int(observations), 0)
    requested_block = max(int(block_length), 1)
    effective_blocks = float(n) / float(requested_block) if n else 0.0
    reasons: list[str] = []
    if n < 2:
        reasons.append("fewer_than_two_observations")
    if requested_block >= n and n > 0:
        reasons.append("block_length_not_smaller_than_sample")
    if effective_blocks < float(minimum_effective_blocks):
        reasons.append("too_few_effective_blocks")
    return {
        "status": "available" if not reasons else "not_available",
        "eligible_for_confirmation": not reasons,
        "nominal_observations": n,
        "requested_block_length": requested_block,
        "effective_blocks": effective_blocks,
        "minimum_effective_blocks": float(minimum_effective_blocks),
        "unavailable_reasons": reasons,
    }


def audit_bootstrap_distribution(
    statistics: Iterable[float],
    *,
    minimum_unique_statistics: int = _MINIMUM_BOOTSTRAP_UNIQUE_STATISTICS,
) -> dict[str, Any]:
    array = finite_values(statistics)
    if array.size == 0:
        return {
            "status": "not_available",
            "eligible_for_confirmation": False,
            "finite_statistics": 0,
            "unique_statistics": 0,
            "minimum_unique_statistics": int(minimum_unique_statistics),
            "numeric_span": None,
            "unavailable_reasons": ["no_finite_bootstrap_statistics"],
        }

    scale = max(1.0, float(np.max(np.abs(array))))
    tolerance = float(np.finfo(float).eps * scale * 64.0)
    quantized = np.rint(array / tolerance).astype(np.int64)
    unique_count = int(np.unique(quantized).size)
    numeric_span = float(np.ptp(array))
    reasons: list[str] = []
    if numeric_span <= tolerance:
        reasons.append("numerically_degenerate_bootstrap_distribution")
    if unique_count < int(minimum_unique_statistics):
        reasons.append("too_few_unique_bootstrap_statistics")
    return {
        "status": "available" if not reasons else "not_available",
        "eligible_for_confirmation": not reasons,
        "finite_statistics": int(array.size),
        "unique_statistics": unique_count,
        "minimum_unique_statistics": int(minimum_unique_statistics),
        "numeric_span": numeric_span,
        "numeric_tolerance": tolerance,
        "unavailable_reasons": reasons,
    }


def circular_block_bootstrap_mean(
    values: Iterable[float],
    *,
    block_length: int,
    resamples: int = 1000,
    seed: int = 1729,
    confidence: float = 0.95,
) -> dict[str, Any]:
    array = finite_values(values)
    n = int(array.size)
    requested_block = max(1, int(block_length))
    block = max(1, min(requested_block, max(n, 1)))
    design_audit = audit_block_bootstrap_design(
        observations=n,
        block_length=requested_block,
    )
    result = {
        "method": "circular_moving_block_bootstrap",
        "status": "not_available",
        "eligible_for_confirmation": False,
        "nominal_observations": n,
        "requested_block_length": requested_block,
        "block_length": block,
        "design_audit": design_audit,
        "distribution_audit": None,
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "mean": None,
        "confidence_interval_low": None,
        "confidence_interval_high": None,
        "positive_one_sided_p_value": None,
        "negative_one_sided_p_value": None,
        "two_sided_p_value": None,
        "unavailable_reasons": list(design_audit["unavailable_reasons"]),
    }
    if int(resamples) <= 0:
        result["unavailable_reasons"].append("nonpositive_resample_count")
        return result
    if not design_audit["eligible_for_confirmation"]:
        return result
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between zero and one")

    generator = np.random.default_rng(int(seed))
    observed = float(array.mean())
    centered = array - observed
    bootstrap_means = np.empty(int(resamples), dtype=float)
    null_means = np.empty(int(resamples), dtype=float)
    blocks_needed = int(math.ceil(float(n) / float(block)))
    offsets = np.arange(block)
    for index in range(int(resamples)):
        starts = generator.integers(0, n, size=blocks_needed)
        sample_indexes = (
            starts[:, None] + offsets[None, :]
        ).reshape(-1)[:n] % n
        bootstrap_means[index] = float(array[sample_indexes].mean())
        null_means[index] = float(centered[sample_indexes].mean())

    distribution_audit = audit_bootstrap_distribution(null_means)
    result["distribution_audit"] = distribution_audit
    if not distribution_audit["eligible_for_confirmation"]:
        result["unavailable_reasons"] = list(
            distribution_audit["unavailable_reasons"]
        )
        return result

    tail = (1.0 - float(confidence)) / 2.0
    positive = float(
        (1 + np.count_nonzero(null_means >= observed))
        / (int(resamples) + 1)
    )
    negative = float(
        (1 + np.count_nonzero(null_means <= observed))
        / (int(resamples) + 1)
    )
    return {
        **result,
        "status": "available",
        "eligible_for_confirmation": True,
        "mean": observed,
        "confidence_interval_low": float(np.quantile(bootstrap_means, tail)),
        "confidence_interval_high": float(
            np.quantile(bootstrap_means, 1.0 - tail)
        ),
        "positive_one_sided_p_value": positive,
        "negative_one_sided_p_value": negative,
        "two_sided_p_value": min(1.0, 2.0 * min(positive, negative)),
        "unavailable_reasons": [],
    }


def dependence_aware_mean_inference(
    values: Iterable[float],
    *,
    overlap_horizon: int = 1,
    minimum_hac_lag: int = 0,
    block_length: int | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    array = finite_values(values)
    hac = newey_west_mean_inference(
        array,
        overlap_horizon=overlap_horizon,
        minimum_lag=minimum_hac_lag,
    )
    chosen_block = (
        int(block_length)
        if block_length is not None
        else max(int(hac["hac_lag"]) + 1, int(overlap_horizon), 1)
    )
    bootstrap = circular_block_bootstrap_mean(
        array,
        block_length=chosen_block,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    inference_available = bool(
        hac["hac_t_stat"] is not None
        and bootstrap["eligible_for_confirmation"]
    )
    return {
        "method": "date_level_newey_west_and_circular_block_bootstrap",
        "overlap_horizon": int(overlap_horizon),
        "hac": hac,
        "block_bootstrap": bootstrap,
        "confirmation_inference_available": inference_available,
        "confirmation_inference_unavailable_reasons": (
            []
            if inference_available
            else (
                list(bootstrap["unavailable_reasons"])
                if hac["hac_t_stat"] is not None
                else ["hac_mean_inference_not_available"]
                + list(bootstrap["unavailable_reasons"])
            )
        ),
        "iid_inference_eligible": False,
    }


def hac_t_statistic(
    values: Iterable[float],
    *,
    overlap_horizon: int = 1,
    minimum_lag: int = 0,
) -> float:
    result = newey_west_mean_inference(
        values,
        overlap_horizon=overlap_horizon,
        minimum_lag=minimum_lag,
    )
    value = result["hac_t_stat"]
    return float(value) if value is not None else float("nan")


def hac_adjusted_annualized_ratio(
    values: Iterable[float],
    *,
    periods_per_year: int = 252,
    overlap_horizon: int = 1,
    minimum_lag: int = 0,
) -> float:
    result = newey_west_mean_inference(
        values,
        overlap_horizon=overlap_horizon,
        minimum_lag=minimum_lag,
    )
    mean = result["mean"]
    long_run_variance = result["hac_long_run_variance"]
    if mean is None or long_run_variance is None or long_run_variance <= 1e-15:
        return float("nan")
    return float(mean / math.sqrt(long_run_variance) * math.sqrt(periods_per_year))
