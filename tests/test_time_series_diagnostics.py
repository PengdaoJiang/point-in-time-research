from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qplatform.time_series_diagnostics import (
    benjamini_hochberg,
    build_serial_dependence_diagnostics,
    build_series_diagnostic,
    chi_square_survival,
)


def test_chi_square_survival_known_even_degrees() -> None:
    statistic = 4.0
    assert math.isclose(
        chi_square_survival(statistic, 2),
        math.exp(-statistic / 2.0),
        rel_tol=1e-12,
    )
    assert math.isclose(
        chi_square_survival(statistic, 4),
        math.exp(-statistic / 2.0) * (1.0 + statistic / 2.0),
        rel_tol=1e-12,
    )


def test_benjamini_hochberg_preserves_order_and_monotonicity() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, None])
    assert adjusted == [0.03, 0.04, 0.04, None]


def test_ar1_has_linear_memory_and_dominant_first_pacf() -> None:
    generator = np.random.default_rng(1729)
    values = np.empty(4000, dtype=float)
    values[0] = generator.normal()
    for index in range(1, values.size):
        values[index] = 0.72 * values[index - 1] + generator.normal()
    diagnostic = build_series_diagnostic(values, series_name="ar1", max_lag=12)
    assert diagnostic["lags"][0]["acf"] > 0.65
    assert diagnostic["lags"][0]["pacf"] > 0.65
    assert max(abs(row["pacf"]) for row in diagnostic["lags"][1:]) < 0.08
    assert diagnostic["ljung_box"]["reject_no_linear_autocorrelation"] is True


def test_seeded_white_noise_does_not_trigger_linear_memory_diagnostic() -> None:
    values = np.random.default_rng(1729).normal(size=5000)
    diagnostic = build_series_diagnostic(
        values,
        series_name="white_noise",
        max_lag=12,
    )
    assert diagnostic["any_acf_lag_bh_significant"] is False
    assert diagnostic["ljung_box"]["reject_no_linear_autocorrelation"] is False
    assert (
        diagnostic["ljung_box"][
            "failure_to_reject_proves_independence_or_iid"
        ]
        is False
    )


def test_ma1_acf_cuts_off_after_first_lag_in_population_pattern() -> None:
    generator = np.random.default_rng(991)
    innovations = generator.normal(size=6001)
    values = innovations[1:] + 0.8 * innovations[:-1]
    diagnostic = build_series_diagnostic(values, series_name="ma1", max_lag=10)
    assert diagnostic["lags"][0]["acf"] > 0.45
    assert max(abs(row["acf"]) for row in diagnostic["lags"][1:]) < 0.06


def test_volatility_memory_can_exist_without_directional_memory() -> None:
    generator = np.random.default_rng(2718)
    log_variance = np.empty(6000, dtype=float)
    log_variance[0] = -2.0
    for index in range(1, log_variance.size):
        log_variance[index] = (
            -0.1 + 0.95 * log_variance[index - 1] + 0.18 * generator.normal()
        )
    returns = np.exp(0.5 * log_variance) * generator.normal(size=log_variance.size)
    result = build_serial_dependence_diagnostics(returns, max_lag=12)
    assert abs(result["series"]["returns"]["lags"][0]["acf"]) < 0.05
    assert result["summary"]["volatility_memory_detected"] is True
    assert result["interpretation"]["zero_return_acf_proves_unpredictable"] is False
    assert result["promotion_authority"] is False


def test_non_finite_series_fails_closed() -> None:
    try:
        build_series_diagnostic([0.1, float("nan"), 0.2], series_name="bad")
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite time-series observations must fail closed")


if __name__ == "__main__":
    test_chi_square_survival_known_even_degrees()
    test_benjamini_hochberg_preserves_order_and_monotonicity()
    test_ar1_has_linear_memory_and_dominant_first_pacf()
    test_seeded_white_noise_does_not_trigger_linear_memory_diagnostic()
    test_ma1_acf_cuts_off_after_first_lag_in_population_pattern()
    test_volatility_memory_can_exist_without_directional_memory()
    test_non_finite_series_fails_closed()
    print("PASS: time-series diagnostics")
