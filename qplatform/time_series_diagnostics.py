from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


_FLOAT_EPSILON = float(np.finfo(float).eps)
_FLOAT_TINY = float(np.finfo(float).tiny)


def validated_time_series(values: Iterable[float], *, name: str) -> np.ndarray:
    """Return a finite ordered series without silently collapsing missing periods."""
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite observations")
    return array


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    adjusted: list[float | None] = [None] * len(values)
    if not valid:
        return adjusted

    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        original_index, value = ordered[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, value * count / rank)
        adjusted[original_index] = float(np.clip(running, 0.0, 1.0))
    return adjusted


def _regularized_gamma_q(shape: float, argument: float) -> float:
    """Regularized upper incomplete gamma for chi-square tail probabilities."""
    if shape <= 0.0 or argument < 0.0:
        raise ValueError("gamma arguments are outside the supported domain")
    if argument == 0.0:
        return 1.0

    iterations = 1000
    tolerance = 3e-14
    floor = 1e-300
    log_scale = -argument + shape * math.log(argument) - math.lgamma(shape)

    if argument < shape + 1.0:
        term = 1.0 / shape
        total = term
        denominator = shape
        for _ in range(iterations):
            denominator += 1.0
            term *= argument / denominator
            total += term
            if abs(term) <= abs(total) * tolerance:
                lower = total * math.exp(log_scale)
                return float(np.clip(1.0 - lower, 0.0, 1.0))
        raise ArithmeticError("regularized gamma series did not converge")

    b_value = argument + 1.0 - shape
    c_value = 1.0 / floor
    d_value = 1.0 / max(abs(b_value), floor)
    if b_value < 0.0:
        d_value = -d_value
    fraction = d_value
    for index in range(1, iterations + 1):
        coefficient = -float(index) * (float(index) - shape)
        b_value += 2.0
        d_value = coefficient * d_value + b_value
        if abs(d_value) < floor:
            d_value = floor
        c_value = b_value + coefficient / c_value
        if abs(c_value) < floor:
            c_value = floor
        d_value = 1.0 / d_value
        delta = d_value * c_value
        fraction *= delta
        if abs(delta - 1.0) <= tolerance:
            return float(np.clip(math.exp(log_scale) * fraction, 0.0, 1.0))
    raise ArithmeticError("regularized gamma continued fraction did not converge")


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    if statistic < 0.0:
        raise ValueError("chi-square statistic must be non-negative")
    if int(degrees_of_freedom) <= 0:
        raise ValueError("chi-square degrees of freedom must be positive")
    return _regularized_gamma_q(
        float(degrees_of_freedom) / 2.0,
        float(statistic) / 2.0,
    )


def _evaluated_max_lag(observations: int, requested_max_lag: int) -> int:
    requested = int(requested_max_lag)
    if requested < 1:
        raise ValueError("max_lag must be positive")
    if observations < 4:
        return 0
    # PACF uses an expanding lag regression. The n/4 cap avoids nearly
    # saturated regressions while preserving the caller's predeclared limit.
    return min(requested, observations // 4, observations - 2)


def _has_resolvable_variation(array: np.ndarray) -> bool:
    if array.size == 0:
        return False
    centered = array - float(array.mean())
    centered_energy = float(np.dot(centered, centered))
    scale_energy = max(float(np.dot(array, array)), _FLOAT_TINY)
    return centered_energy > _FLOAT_EPSILON * scale_energy


def autocorrelations(values: Iterable[float], *, max_lag: int) -> list[float | None]:
    array = validated_time_series(values, name="time series")
    evaluated = _evaluated_max_lag(int(array.size), int(max_lag))
    if evaluated == 0:
        return []
    centered = array - float(array.mean())
    denominator = float(np.dot(centered, centered))
    if not _has_resolvable_variation(array):
        return [None] * evaluated
    return [
        float(np.dot(centered[lag:], centered[:-lag]) / denominator)
        for lag in range(1, evaluated + 1)
    ]


def partial_autocorrelations(
    values: Iterable[float],
    *,
    max_lag: int,
) -> list[float | None]:
    array = validated_time_series(values, name="time series")
    evaluated = _evaluated_max_lag(int(array.size), int(max_lag))
    if evaluated == 0:
        return []
    centered = array - float(array.mean())
    if not _has_resolvable_variation(array):
        return [None] * evaluated

    output: list[float | None] = []
    observations = int(centered.size)
    for lag in range(1, evaluated + 1):
        target = centered[lag:]
        design = np.column_stack(
            [
                np.ones(observations - lag),
                *[
                    centered[lag - offset : observations - offset]
                    for offset in range(1, lag + 1)
                ],
            ]
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        output.append(float(coefficients[-1]) if rank == design.shape[1] else None)
    return output


def _pointwise_p_value(correlation: float | None, observations: int) -> float | None:
    if correlation is None or observations <= 0:
        return None
    statistic = abs(float(correlation)) * math.sqrt(float(observations))
    return float(math.erfc(statistic / math.sqrt(2.0)))


def build_series_diagnostic(
    values: Iterable[float],
    *,
    series_name: str,
    max_lag: int = 20,
    model_degrees_of_freedom: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    array = validated_time_series(values, name=series_name)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    if int(model_degrees_of_freedom) < 0:
        raise ValueError("model_degrees_of_freedom must be non-negative")

    observations = int(array.size)
    evaluated = _evaluated_max_lag(observations, int(max_lag))
    acf = autocorrelations(array, max_lag=max_lag) if evaluated else []
    pacf = partial_autocorrelations(array, max_lag=max_lag) if evaluated else []
    acf_p = [_pointwise_p_value(value, observations) for value in acf]
    pacf_p = [_pointwise_p_value(value, observations) for value in pacf]
    acf_q = benjamini_hochberg(acf_p)
    pacf_q = benjamini_hochberg(pacf_p)
    pointwise_band = (
        1.959963984540054 / math.sqrt(float(observations))
        if observations > 0
        else None
    )

    rows: list[dict[str, Any]] = []
    for index in range(evaluated):
        rows.append(
            {
                "lag": index + 1,
                "acf": acf[index],
                "acf_pointwise_p_value_approx": acf_p[index],
                "acf_bh_q_value": acf_q[index],
                "acf_exceeds_pointwise_95pct_band": (
                    abs(float(acf[index])) > float(pointwise_band)
                    if acf[index] is not None and pointwise_band is not None
                    else None
                ),
                "pacf": pacf[index],
                "pacf_pointwise_p_value_approx": pacf_p[index],
                "pacf_bh_q_value": pacf_q[index],
                "pacf_exceeds_pointwise_95pct_band": (
                    abs(float(pacf[index])) > float(pointwise_band)
                    if pacf[index] is not None and pointwise_band is not None
                    else None
                ),
            }
        )

    usable_acf = [(index + 1, value) for index, value in enumerate(acf) if value is not None]
    strongest = max(usable_acf, key=lambda item: abs(item[1])) if usable_acf else None
    q_statistic = None
    q_degrees = evaluated - int(model_degrees_of_freedom)
    q_p_value = None
    if evaluated and usable_acf:
        q_statistic = float(
            observations
            * (observations + 2)
            * sum(
                float(value) ** 2 / float(observations - lag)
                for lag, value in usable_acf
            )
        )
        if q_degrees > 0:
            q_p_value = chi_square_survival(q_statistic, q_degrees)

    any_acf_bh = any(value is not None and value <= alpha for value in acf_q)
    any_pacf_bh = any(value is not None and value <= alpha for value in pacf_q)
    joint_rejection = q_p_value is not None and q_p_value <= alpha
    return {
        "series_name": str(series_name),
        "status": (
            "computed"
            if evaluated and usable_acf
            else "insufficient_or_constant_series"
        ),
        "nominal_observations": observations,
        "requested_max_lag": int(max_lag),
        "evaluated_max_lag": evaluated,
        "lag_cap_rule": "min(predeclared_max_lag,floor(n/4),n-2)",
        "model_degrees_of_freedom": int(model_degrees_of_freedom),
        "alpha": float(alpha),
        "pointwise_95pct_band_approx": pointwise_band,
        "constant_series": bool(observations and not _has_resolvable_variation(array)),
        "zero_fraction": float(np.mean(array == 0.0)) if observations else None,
        "strongest_absolute_acf_lag": strongest[0] if strongest else None,
        "strongest_absolute_acf": strongest[1] if strongest else None,
        "any_acf_lag_bh_significant": any_acf_bh,
        "any_pacf_lag_bh_significant": any_pacf_bh,
        "ljung_box": {
            "statistic": q_statistic,
            "degrees_of_freedom": q_degrees if q_degrees > 0 else None,
            "p_value": q_p_value,
            "reject_no_linear_autocorrelation": joint_rejection,
            "null": "no_linear_autocorrelation_through_evaluated_max_lag",
            "heteroskedasticity_robust": False,
            "failure_to_reject_proves_independence_or_iid": False,
        },
        "lags": rows,
        "inference_scope": (
            "ACF, PACF, and Ljung-Box diagnose linear serial dependence only; "
            "pointwise p-values and bands are asymptotic diagnostics, while BH "
            "q-values adjust the inspected lag family."
        ),
        "promotion_authority": False,
    }


def build_serial_dependence_diagnostics(
    returns: Iterable[float],
    *,
    max_lag: int = 20,
    residuals: Iterable[float] | None = None,
    residual_model_degrees_of_freedom: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    return_array = validated_time_series(returns, name="returns")
    diagnostics = {
        "returns": build_series_diagnostic(
            return_array,
            series_name="returns",
            max_lag=max_lag,
            alpha=alpha,
        ),
        "absolute_returns": build_series_diagnostic(
            np.abs(return_array),
            series_name="absolute_returns",
            max_lag=max_lag,
            alpha=alpha,
        ),
        "squared_returns": build_series_diagnostic(
            np.square(return_array),
            series_name="squared_returns",
            max_lag=max_lag,
            alpha=alpha,
        ),
    }
    if residuals is not None:
        diagnostics["model_residuals"] = build_series_diagnostic(
            residuals,
            series_name="model_residuals",
            max_lag=max_lag,
            model_degrees_of_freedom=residual_model_degrees_of_freedom,
            alpha=alpha,
        )

    def detected(name: str) -> bool:
        diagnostic = diagnostics[name]
        return bool(
            diagnostic["any_acf_lag_bh_significant"]
            or diagnostic["ljung_box"]["reject_no_linear_autocorrelation"]
        )

    return {
        "schema_version": 1,
        "method": "acf_pacf_ljung_box_with_bh_lag_family_adjustment",
        "predeclared_max_lag": int(max_lag),
        "alpha": float(alpha),
        "series": diagnostics,
        "summary": {
            "linear_return_memory_detected": detected("returns"),
            "volatility_memory_detected": (
                detected("absolute_returns") or detected("squared_returns")
            ),
            "model_residual_linear_memory_detected": (
                detected("model_residuals")
                if "model_residuals" in diagnostics
                else None
            ),
        },
        "interpretation": {
            "zero_return_acf_proves_unpredictable": False,
            "acf_pacf_are_general_predictability_tests": False,
            "ljung_box_failure_to_reject_proves_independence_or_iid": False,
            "volatility_memory_implies_profitable_directional_rule": False,
            "full_sample_order_selection_is_confirmation_evidence": False,
            "required_use": (
                "Use returns rather than raw price levels for ordinary memory "
                "diagnostics; inspect model residuals when available; freeze any "
                "lag or order selected in discovery before causal multi-origin "
                "confirmation."
            ),
        },
        "promotion_authority": False,
    }
