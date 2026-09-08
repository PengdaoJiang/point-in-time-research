from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qplatform.dependence_statistics import (
    audit_block_bootstrap_design,
    audit_bootstrap_distribution,
    circular_block_bootstrap_mean,
    dependence_aware_mean_inference,
    newey_west_mean_inference,
    recommended_hac_lag,
)


def test_autocorrelation_reduces_effective_history() -> None:
    generator = np.random.default_rng(20260808)
    innovations = generator.normal(0.0, 1.0, size=600)
    values = np.empty_like(innovations)
    values[0] = innovations[0]
    for index in range(1, values.size):
        values[index] = 0.85 * values[index - 1] + innovations[index]
    values = values + 0.15

    result = newey_west_mean_inference(values)
    assert result["hac_standard_error"] > result["iid_standard_error_diagnostic"]
    assert 1.0 <= result["effective_observations"] < values.size * 0.6
    assert result["hac_t_stat"] < result["iid_t_stat_diagnostic"]


def test_overlap_horizon_sets_inference_floor() -> None:
    assert recommended_hac_lag(200, overlap_horizon=12) >= 11
    result = dependence_aware_mean_inference(
        np.linspace(-0.02, 0.03, 200),
        overlap_horizon=12,
        bootstrap_resamples=100,
    )
    assert result["hac"]["hac_lag"] >= 11
    assert result["block_bootstrap"]["block_length"] >= 12
    assert result["confirmation_inference_available"] is True


def test_circular_block_bootstrap_is_deterministic() -> None:
    values = np.sin(np.linspace(0.0, 8.0, 120)) * 0.01 + 0.001
    first = circular_block_bootstrap_mean(
        values,
        block_length=7,
        resamples=200,
        seed=41,
    )
    second = circular_block_bootstrap_mean(
        values,
        block_length=7,
        resamples=200,
        seed=41,
    )
    assert first == second
    assert first["confidence_interval_low"] < first["confidence_interval_high"]


def test_full_sample_block_bootstrap_is_not_available() -> None:
    values = np.linspace(-0.01, 0.02, 12)
    result = circular_block_bootstrap_mean(
        values,
        block_length=12,
        resamples=200,
        seed=41,
    )
    assert result["status"] == "not_available"
    assert result["eligible_for_confirmation"] is False
    assert result["two_sided_p_value"] is None
    assert "block_length_not_smaller_than_sample" in result["unavailable_reasons"]
    assert "too_few_effective_blocks" in result["unavailable_reasons"]

    combined = dependence_aware_mean_inference(
        values,
        block_length=12,
        bootstrap_resamples=200,
    )
    assert combined["confirmation_inference_available"] is False


def test_degenerate_bootstrap_distribution_is_not_available() -> None:
    design = audit_block_bootstrap_design(observations=40, block_length=5)
    assert design["eligible_for_confirmation"] is True
    distribution = audit_bootstrap_distribution([0.0] * 1000)
    assert distribution["status"] == "not_available"
    assert "numerically_degenerate_bootstrap_distribution" in distribution[
        "unavailable_reasons"
    ]

    result = circular_block_bootstrap_mean(
        [0.001] * 40,
        block_length=5,
        resamples=200,
        seed=41,
    )
    assert result["status"] == "not_available"
    assert result["confidence_interval_low"] is None
    assert result["positive_one_sided_p_value"] is None






