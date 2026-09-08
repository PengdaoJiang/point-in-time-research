from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np
import pandas as pd


_FORBIDDEN_FEATURE_TOKENS = {
    "entry",
    "exit",
    "future",
    "label",
    "lead",
    "target",
}

_ADAPTIVE_UPDATE_CLASSES = {
    "scheduled_parameter_refit",
    "prevalidated_library_switch",
    "new_formula_or_structure_discovery",
}

_ADAPTIVE_TRIGGER_CLASSES = {
    "scheduled_calendar",
    "data_advance",
    "explicit_research_program",
    "monitoring_threshold",
    "user_nominated_hypothesis",
    "prospective_monitoring",
}

_ADAPTIVE_EVIDENCE_CLASSES = {
    "scheduled_parameter_refit_snapshot",
    "adaptive_discovery_reused_outcomes",
    "causal_confirmation_reused_outcomes",
    "fresh_independent_confirmation",
    "mixed_reused_and_independent_confirmation",
    "prospective_monitoring",
}

_RESEARCH_EVIDENCE_STAGES = {
    "discovery",
    "confirmation",
    "promotion_audit",
}
_RESEARCH_EVIDENCE_STATUSES = {
    "pass",
    "partial",
    "not_available",
    "not_applicable",
    "fail",
}
_ATTRIBUTION_DIMENSIONS = {
    "market_beta",
    "industry",
    "size",
    "value",
    "momentum",
}

_RESEARCH_ACCOUNT_MODE = "isolated_synthetic_fixed_capital"
_DEFAULT_RESEARCH_INITIAL_CASH_CNY = 150_000.0
_FORBIDDEN_RESEARCH_ACCOUNT_INPUT_FILES = {
    "account_scope.json",
    "current_holdings.csv",
    "current_risk_limits.json",
    "operator_trade_ledger.csv",
    "operator_trade_ledger_meta.json",
}
_FORBIDDEN_RESEARCH_ACCOUNT_INPUT_FIELDS = {
    "available_margin_cny",
    "capital_basis_cny",
    "cash_available_cny",
    "controlled_position_market_value_cny",
    "controlled_symbols",
    "financing_payable_cny",
    "project_controllable_notional_cny",
    "risk_capital_basis_cny",
}


def _feature_tokens(name: str) -> set[str]:
    normalized = str(name or "").strip().lower().replace("-", "_")
    return {token for token in normalized.split("_") if token}


def _is_forbidden_research_feature(name: str) -> bool:
    normalized = str(name or "").strip().lower().replace("-", "_")
    return bool(
        _feature_tokens(normalized).intersection(_FORBIDDEN_FEATURE_TOKENS)
        or normalized in _FORBIDDEN_RESEARCH_ACCOUNT_INPUT_FIELDS
    )


def audit_research_feature_columns(feature_columns: Iterable[str]) -> dict[str, Any]:
    columns = [str(column) for column in feature_columns]
    forbidden = sorted(
        {
            column
            for column in columns
            if _is_forbidden_research_feature(column)
        }
    )
    duplicates = sorted(
        column for column, count in Counter(columns).items() if count > 1
    )
    return {
        "feature_count": int(len(columns)),
        "allowed_feature_count": int(len(columns) - len(forbidden)),
        "forbidden_feature_count": int(len(forbidden)),
        "forbidden_features": forbidden,
        "duplicate_features": duplicates,
        "pass": not forbidden and not duplicates,
    }


def assert_research_feature_columns(feature_columns: Iterable[str]) -> list[str]:
    columns = [str(column) for column in feature_columns]
    audit = audit_research_feature_columns(columns)
    if not audit["pass"]:
        raise ValueError(
            "research feature guard failed: "
            f"forbidden={audit['forbidden_features']} "
            f"duplicates={audit['duplicate_features']}"
        )
    return columns


def allowed_research_feature_columns(feature_columns: Iterable[str]) -> list[str]:
    columns = sorted({str(column) for column in feature_columns})
    return [
        column
        for column in columns
        if not _is_forbidden_research_feature(column)
    ]


def audit_research_account_isolation(
    payload: Mapping[str, Any],
    *,
    expected_initial_cash_cny: float = _DEFAULT_RESEARCH_INITIAL_CASH_CNY,
) -> dict[str, Any]:
    """Reject real broker-account occupancy from research and training inputs."""

    errors: list[str] = []
    mode = str(payload.get("mode") or "").strip()
    if mode != _RESEARCH_ACCOUNT_MODE:
        errors.append("research_account_mode_must_be_isolated_synthetic_fixed_capital")

    initial_cash_raw = payload.get("initial_cash_cny")
    try:
        initial_cash_cny = float(initial_cash_raw)
    except (TypeError, ValueError):
        initial_cash_cny = float("nan")
        errors.append("initial_cash_cny_must_be_numeric")
    if np.isfinite(initial_cash_cny):
        if initial_cash_cny <= 0:
            errors.append("initial_cash_cny_must_be_positive")
        elif not math.isclose(
            initial_cash_cny,
            float(expected_initial_cash_cny),
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            errors.append("initial_cash_cny_must_match_canonical_research_capital")

    initial_inventory_mode = str(
        payload.get("initial_inventory_mode") or ""
    ).strip()
    if initial_inventory_mode not in {"flat", "predeclared_synthetic"}:
        errors.append("initial_inventory_mode_must_be_flat_or_predeclared_synthetic")

    required_false = [
        "real_account_holdings_or_positions_used",
        "real_account_cash_margin_or_financing_used",
        "real_account_risk_limits_used",
        "real_account_transactions_used_as_training_rows_labels_or_weights",
        "real_account_state_used_for_candidate_selection_or_rejection",
    ]
    required_true = [
        "synthetic_strategy_occupancy_enforced",
        "synthetic_shared_cash_ledger_enforced",
        "post_research_execution_overlay_separated",
    ]
    for field in required_false + required_true:
        if not isinstance(payload.get(field), bool):
            errors.append(f"{field}_must_be_boolean")
    for field in required_false:
        if payload.get(field) is True:
            errors.append(f"{field}_must_be_false")
    for field in required_true:
        if payload.get(field) is False:
            errors.append(f"{field}_must_be_true")

    paths_raw = payload.get("research_input_paths")
    if not isinstance(paths_raw, list):
        errors.append("research_input_paths_must_be_a_list")
        paths_raw = []
    normalized_paths: list[str] = []
    forbidden_paths: list[str] = []
    for value in paths_raw:
        normalized = str(value or "").strip().replace("\\", "/").lower()
        if not normalized:
            errors.append("research_input_paths_must_not_contain_blank_values")
            continue
        normalized_paths.append(normalized)
        basename = normalized.rsplit("/", 1)[-1]
        if (
            "/workspace_out/state/account/" in f"/{normalized.lstrip('/')}"
            or basename in _FORBIDDEN_RESEARCH_ACCOUNT_INPUT_FILES
        ):
            forbidden_paths.append(str(value))
    if forbidden_paths:
        errors.append("real_account_state_path_used_as_research_input")

    fields_raw = payload.get("research_input_fields")
    if not isinstance(fields_raw, list):
        errors.append("research_input_fields_must_be_a_list")
        fields_raw = []
    normalized_fields = {
        str(value or "").strip().lower()
        for value in fields_raw
        if str(value or "").strip()
    }
    forbidden_fields = sorted(
        normalized_fields.intersection(_FORBIDDEN_RESEARCH_ACCOUNT_INPUT_FIELDS)
    )
    if forbidden_fields:
        errors.append("real_account_occupancy_field_used_as_research_input")

    return {
        "pass": not errors,
        "errors": sorted(set(errors)),
        "mode": mode,
        "initial_cash_cny": (
            initial_cash_cny if np.isfinite(initial_cash_cny) else None
        ),
        "expected_initial_cash_cny": float(expected_initial_cash_cny),
        "initial_inventory_mode": initial_inventory_mode,
        "forbidden_research_input_paths": sorted(set(forbidden_paths)),
        "forbidden_research_input_fields": forbidden_fields,
        "research_input_path_count": len(normalized_paths),
        "contract": {
            "strategy_generated_synthetic_occupancy_required": True,
            "user_real_account_occupancy_allowed_in_research": False,
            "real_account_state_role": (
                "post_research_execution_feasibility_and_reconciliation_only"
            ),
            "operator_case_exception": (
                "explicit_hypothesis_generation_only_never_training_label_weight_"
                "or_backtest_initial_state"
            ),
        },
    }


def assert_research_account_isolation(
    payload: Mapping[str, Any],
    *,
    expected_initial_cash_cny: float = _DEFAULT_RESEARCH_INITIAL_CASH_CNY,
) -> dict[str, Any]:
    audit = audit_research_account_isolation(
        payload,
        expected_initial_cash_cny=expected_initial_cash_cny,
    )
    if not audit["pass"]:
        raise ValueError(f"research account isolation guard failed: {audit['errors']}")
    return audit


def cross_section_rank_score(
    frame: pd.DataFrame,
    *,
    value_col: str,
    date_col: str = "date",
    higher_is_better: bool,
) -> pd.Series:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    ranks = values.groupby(frame[date_col]).rank(
        method="average",
        pct=True,
        ascending=bool(higher_is_better),
    )
    return ranks.fillna(0.5) - 0.5


def audit_origin_outcome_returns(
    frame: pd.DataFrame,
    *,
    return_col: str,
    resolved_col: str,
    action_col: str,
    cash_actions: Iterable[str] = ("ABSTAIN", "CASH", "NO_ACTION"),
    zero_tolerance: float = 1e-15,
) -> dict[str, Any]:
    required = [return_col, resolved_col, action_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return {
            "pass": False,
            "rows": int(frame.shape[0]),
            "missing_columns": missing,
            "errors": ["missing_required_columns"],
        }

    resolved_raw = frame[resolved_col]
    resolved = resolved_raw.map(
        lambda value: (
            value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
    )
    valid_resolution = resolved_raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
        or str(value).strip().lower() in {"0", "1", "false", "true", "no", "yes"}
    )
    actions = frame[action_col].astype("string").fillna("").str.strip().str.upper()
    cash_action_set = {str(value).strip().upper() for value in cash_actions}
    is_cash = actions.isin(cash_action_set)
    returns = pd.to_numeric(frame[return_col], errors="coerce")
    finite = pd.Series(np.isfinite(returns.to_numpy(dtype=float)), index=frame.index)
    nonzero = finite & returns.abs().gt(float(zero_tolerance))

    invalid_resolution_count = int((~valid_resolution).sum())
    resolved_cash_nonfinite_count = int((resolved & is_cash & ~finite).sum())
    resolved_cash_nonzero_count = int((resolved & is_cash & nonzero).sum())
    resolved_position_nonfinite_count = int((resolved & ~is_cash & ~finite).sum())
    unresolved_finite_count = int((~resolved & finite).sum())
    errors: list[str] = []
    if invalid_resolution_count:
        errors.append("invalid_resolution_value")
    if resolved_cash_nonfinite_count:
        errors.append("resolved_cash_return_must_be_explicit_zero")
    if resolved_cash_nonzero_count:
        errors.append("resolved_cash_return_must_equal_zero")
    if resolved_position_nonfinite_count:
        errors.append("resolved_position_return_must_be_finite")
    if unresolved_finite_count:
        errors.append("unresolved_outcome_return_must_remain_missing")

    return {
        "pass": not errors,
        "rows": int(frame.shape[0]),
        "missing_columns": [],
        "resolved_rows": int(resolved.sum()),
        "unresolved_rows": int((~resolved).sum()),
        "resolved_cash_rows": int((resolved & is_cash).sum()),
        "invalid_resolution_count": invalid_resolution_count,
        "resolved_cash_nonfinite_count": resolved_cash_nonfinite_count,
        "resolved_cash_nonzero_count": resolved_cash_nonzero_count,
        "resolved_position_nonfinite_count": resolved_position_nonfinite_count,
        "unresolved_finite_count": unresolved_finite_count,
        "cash_actions": sorted(cash_action_set),
        "errors": errors,
        "contract": {
            "resolved_cash_or_abstention": "explicit_zero",
            "resolved_position": "finite_realized_return",
            "unresolved_fill_or_exit": "missing_never_zero_filled",
        },
    }


def assert_origin_outcome_returns(
    frame: pd.DataFrame,
    *,
    return_col: str,
    resolved_col: str,
    action_col: str,
    cash_actions: Iterable[str] = ("ABSTAIN", "CASH", "NO_ACTION"),
) -> dict[str, Any]:
    audit = audit_origin_outcome_returns(
        frame,
        return_col=return_col,
        resolved_col=resolved_col,
        action_col=action_col,
        cash_actions=cash_actions,
    )
    if not audit["pass"]:
        raise ValueError(f"origin outcome return guard failed: {audit['errors']}")
    return audit


def audit_execution_observability_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that historical fills use only execution-point evidence."""

    required_text = [
        "entry_clock",
        "exit_clock",
        "entry_fill_price_source",
        "exit_fill_price_source",
        "price_limit_policy",
    ]
    errors = [
        f"missing_required_field:{field}"
        for field in required_text
        if not str(payload.get(field) or "").strip()
    ]

    required_booleans = [
        "fill_inputs_available_by_execution_time",
        "post_execution_extrema_used_for_fill",
        "later_unseal_backfills_earlier_fill",
        "blocked_exit_retries_at_same_declared_clock",
        "right_censored_plans_omitted_not_zero_filled",
        "closing_auction_order_predeclared_if_used",
    ]
    booleans: dict[str, bool] = {}
    for field in required_booleans:
        value = payload.get(field)
        if not isinstance(value, bool):
            errors.append(f"{field}_must_be_boolean")
            continue
        booleans[field] = value

    if booleans.get("fill_inputs_available_by_execution_time") is False:
        errors.append("fill_inputs_not_available_by_execution_time")
    if booleans.get("post_execution_extrema_used_for_fill") is True:
        errors.append("post_execution_high_low_must_not_determine_fill")
    if booleans.get("later_unseal_backfills_earlier_fill") is True:
        errors.append("later_unseal_must_not_backfill_earlier_fill")
    if booleans.get("blocked_exit_retries_at_same_declared_clock") is False:
        errors.append("blocked_exit_retry_clock_changed")
    if booleans.get("right_censored_plans_omitted_not_zero_filled") is False:
        errors.append("right_censored_plan_must_not_be_zero_filled")

    clocks_and_sources = " ".join(
        str(payload.get(field) or "").strip().lower()
        for field in (
            "entry_clock",
            "exit_clock",
            "entry_fill_price_source",
            "exit_fill_price_source",
        )
    )
    uses_closing_auction = "closing_auction" in clocks_and_sources
    if uses_closing_auction and not booleans.get(
        "closing_auction_order_predeclared_if_used", False
    ):
        errors.append("closing_auction_order_must_be_predeclared")

    price_limit_policy = str(payload.get("price_limit_policy") or "").strip()
    if price_limit_policy not in {"queue_conservative", "not_applicable"}:
        errors.append("invalid_price_limit_policy")
    if price_limit_policy == "queue_conservative":
        if (
            str(payload.get("upper_limit_buy_without_queue_evidence") or "").strip()
            != "no_fill_cash"
        ):
            errors.append("upper_limit_buy_without_queue_evidence_must_be_no_fill_cash")
        if (
            str(payload.get("lower_limit_sell_without_queue_evidence") or "").strip()
            != "blocked_exit_retry"
        ):
            errors.append(
                "lower_limit_sell_without_queue_evidence_must_be_blocked_exit_retry"
            )
    elif price_limit_policy == "not_applicable" and not str(
        payload.get("price_limit_not_applicable_reason") or ""
    ).strip():
        errors.append("price_limit_not_applicable_reason_required")

    assertions = payload.get("synthetic_invariance_assertions")
    if not isinstance(assertions, Mapping):
        errors.append("synthetic_invariance_assertions_required")
        assertions = {}
    required_assertions = [
        "open_at_upper_later_low_does_not_fill",
        "close_at_lower_earlier_high_does_not_fill",
        "open_exit_at_lower_later_high_does_not_fill",
    ]
    for field in required_assertions:
        if assertions.get(field) is not True:
            errors.append(f"synthetic_assertion_failed:{field}")

    return {
        "pass": not errors,
        "errors": sorted(set(errors)),
        "entry_clock": str(payload.get("entry_clock") or "").strip(),
        "exit_clock": str(payload.get("exit_clock") or "").strip(),
        "price_limit_policy": price_limit_policy,
        "uses_closing_auction": uses_closing_auction,
        "contract": {
            "feature_label_no_future_is_sufficient": False,
            "post_execution_extrema_may_determine_fill": False,
            "later_unseal_may_backfill_earlier_fill": False,
            "queue_evidence_required_for_at_limit_fill": True,
        },
    }


def assert_execution_observability_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    audit = audit_execution_observability_contract(payload)
    if not audit["pass"]:
        raise ValueError(
            f"execution observability guard failed: {audit['errors']}"
        )
    return audit


def audit_adaptive_research_lineage(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate update authority and cross-round trial accounting metadata."""

    required = [
        "update_class",
        "trigger_class",
        "formula_version",
        "formula_or_structure_changed",
        "specification_frozen_before_evaluation",
        "realized_loss_direct_trigger",
        "runtime_mutation_allowed",
        "parent_run_ids",
        "outcome_date_min",
        "outcome_date_max",
        "reused_parent_outcome_dates",
        "reused_outcome_date_count",
        "independent_confirmation_date_count",
        "within_run_declared_trial_count",
        "known_ancestor_trial_count_lower_bound",
        "cumulative_known_trial_count_lower_bound",
        "historical_trial_inventory_complete",
        "evidence_class",
    ]
    missing = [field for field in required if field not in payload]
    errors = [f"missing_required_field:{field}" for field in missing]
    warnings: list[str] = []

    update_class = str(payload.get("update_class") or "").strip()
    trigger_class = str(payload.get("trigger_class") or "").strip()
    evidence_class = str(payload.get("evidence_class") or "").strip()
    formula_version = str(payload.get("formula_version") or "").strip()
    if update_class not in _ADAPTIVE_UPDATE_CLASSES:
        errors.append("invalid_update_class")
    if trigger_class not in _ADAPTIVE_TRIGGER_CLASSES:
        errors.append("invalid_trigger_class")
    if evidence_class not in _ADAPTIVE_EVIDENCE_CLASSES:
        errors.append("invalid_evidence_class")
    if not formula_version:
        errors.append("formula_version_required")

    boolean_fields = [
        "formula_or_structure_changed",
        "specification_frozen_before_evaluation",
        "realized_loss_direct_trigger",
        "runtime_mutation_allowed",
        "reused_parent_outcome_dates",
        "historical_trial_inventory_complete",
    ]
    booleans: dict[str, bool] = {}
    for field in boolean_fields:
        value = payload.get(field)
        if not isinstance(value, bool):
            errors.append(f"{field}_must_be_boolean")
            continue
        booleans[field] = value

    count_fields = [
        "reused_outcome_date_count",
        "independent_confirmation_date_count",
        "within_run_declared_trial_count",
        "known_ancestor_trial_count_lower_bound",
        "cumulative_known_trial_count_lower_bound",
    ]
    counts: dict[str, int] = {}
    for field in count_fields:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            errors.append(f"{field}_must_be_integer")
            continue
        counts[field] = int(value)
        if int(value) < 0:
            errors.append(f"{field}_must_be_nonnegative")
    if counts.get("within_run_declared_trial_count", 0) < 1:
        errors.append("within_run_declared_trial_count_must_be_positive")
    expected_lower_bound = counts.get("within_run_declared_trial_count", 0) + counts.get(
        "known_ancestor_trial_count_lower_bound", 0
    )
    if counts.get("cumulative_known_trial_count_lower_bound", -1) < expected_lower_bound:
        errors.append("cumulative_known_trial_count_lower_bound_too_small")

    parent_run_ids = payload.get("parent_run_ids")
    if not isinstance(parent_run_ids, list) or any(
        not str(value).strip() for value in (parent_run_ids or [])
    ):
        errors.append("parent_run_ids_must_be_a_list_of_nonempty_ids")
        parent_run_ids = []

    for field in ("outcome_date_min", "outcome_date_max"):
        value = str(payload.get(field) or "").strip()
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError):
            errors.append(f"{field}_must_be_a_date")
            continue
        if pd.isna(parsed):
            errors.append(f"{field}_must_be_a_date")
    try:
        if pd.Timestamp(payload.get("outcome_date_min")) > pd.Timestamp(
            payload.get("outcome_date_max")
        ):
            errors.append("outcome_date_range_inverted")
    except (TypeError, ValueError):
        pass

    formula_changed = booleans.get("formula_or_structure_changed")
    frozen = booleans.get("specification_frozen_before_evaluation")
    loss_triggered = booleans.get("realized_loss_direct_trigger")
    runtime_mutation = booleans.get("runtime_mutation_allowed")
    reused = booleans.get("reused_parent_outcome_dates")
    inventory_complete = booleans.get("historical_trial_inventory_complete")

    if frozen is False:
        errors.append("specification_must_be_frozen_before_evaluation")
    if loss_triggered is True:
        errors.append("realized_loss_must_not_directly_trigger_update")
    if runtime_mutation is True:
        errors.append("runtime_formula_mutation_not_allowed")
    if update_class == "scheduled_parameter_refit" and formula_changed is not False:
        errors.append("scheduled_parameter_refit_must_keep_formula_fixed")
    if update_class == "prevalidated_library_switch" and formula_changed is not False:
        errors.append("prevalidated_library_switch_must_keep_library_fixed")
    if (
        update_class == "new_formula_or_structure_discovery"
        and formula_changed is not True
    ):
        errors.append("new_formula_discovery_must_declare_structure_change")

    reused_count = counts.get("reused_outcome_date_count", 0)
    independent_count = counts.get("independent_confirmation_date_count", 0)
    if reused is True:
        if not parent_run_ids:
            errors.append("reused_outcomes_require_parent_run_ids")
        if reused_count < 1:
            errors.append("reused_outcomes_require_positive_date_count")
        if evidence_class == "fresh_independent_confirmation":
            errors.append("reused_outcomes_cannot_be_fresh_independent_evidence")
        warnings.append("reused_outcomes_are_not_fresh_independent_evidence")
    elif reused is False and reused_count != 0:
        errors.append("nonreused_outcomes_must_have_zero_reused_date_count")

    if evidence_class == "fresh_independent_confirmation" and independent_count < 1:
        errors.append("fresh_independent_evidence_requires_confirmation_dates")
    if evidence_class == "mixed_reused_and_independent_confirmation":
        if reused is not True or independent_count < 1:
            errors.append("mixed_evidence_requires_reused_and_independent_dates")
    if inventory_complete is False:
        warnings.append("historical_trial_inventory_incomplete_lower_bound_only")

    return {
        "pass": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "update_class": update_class,
        "trigger_class": trigger_class,
        "evidence_class": evidence_class,
        "reused_parent_outcome_dates": reused,
        "independent_confirmation_date_count": independent_count,
        "cumulative_known_trial_count_lower_bound": counts.get(
            "cumulative_known_trial_count_lower_bound"
        ),
        "historical_trial_inventory_complete": inventory_complete,
        "fresh_independent_evidence": bool(
            evidence_class in {
                "fresh_independent_confirmation",
                "mixed_reused_and_independent_confirmation",
            }
            and independent_count > 0
        ),
        "global_multiplicity_control_claim_allowed": inventory_complete is True,
        "contract": {
            "no_future_or_prequential_implies_independence": False,
            "realized_loss_direct_formula_update_allowed": False,
            "adaptive_reused_history_grants_runtime_authority": False,
        },
    }


def assert_adaptive_research_lineage(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    audit = audit_adaptive_research_lineage(payload)
    if not audit["pass"]:
        raise ValueError(f"adaptive research lineage guard failed: {audit['errors']}")
    return audit


def audit_research_evidence_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit whether a research claim package discloses its full evidence basis."""

    required_sections = [
        "return_drawdown",
        "benchmark",
        "universe",
        "multiplicity",
        "no_future",
        "temporal_stability",
        "attribution",
        "execution",
    ]
    errors: list[str] = []
    warnings: list[str] = []
    blocking_gaps: list[str] = []

    stage = str(payload.get("stage") or "").strip()
    if stage not in _RESEARCH_EVIDENCE_STAGES:
        errors.append("invalid_stage")

    sections: dict[str, Mapping[str, Any]] = {}
    for name in required_sections:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"missing_or_invalid_section:{name}")
            sections[name] = {}
            continue
        sections[name] = value
        status = str(value.get("status") or "").strip()
        if status not in _RESEARCH_EVIDENCE_STATUSES:
            errors.append(f"invalid_status:{name}")
            continue
        if status in {"partial", "not_available", "fail"}:
            reason = str(value.get("gap_reason") or "").strip()
            if not reason:
                errors.append(f"gap_reason_required:{name}")
            if stage in {"confirmation", "promotion_audit"}:
                blocking_gaps.append(name)
        if status == "not_applicable" and not str(
            value.get("not_applicable_reason") or ""
        ).strip():
            errors.append(f"not_applicable_reason_required:{name}")
        if status == "not_applicable" and stage in {
            "confirmation",
            "promotion_audit",
        }:
            blocking_gaps.append(name)

    return_drawdown = sections["return_drawdown"]
    if return_drawdown.get("status") == "pass":
        required_curves = [
            "absolute_equity_curve_path",
            "active_return_equity_curve_path",
            "strategy_to_benchmark_ratio_curve_path",
        ]
        required_drawdowns = [
            "absolute_equity_max_drawdown",
            "active_return_equity_max_drawdown",
            "strategy_to_benchmark_ratio_max_drawdown",
        ]
        for field in required_curves:
            if not str(return_drawdown.get(field) or "").strip():
                errors.append(f"return_drawdown_missing:{field}")
        for field in required_drawdowns:
            value = return_drawdown.get(field)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                errors.append(f"return_drawdown_non_numeric:{field}")
            elif not math.isfinite(float(value)) or float(value) > 0.0:
                errors.append(f"return_drawdown_invalid:{field}")

    benchmark = sections["benchmark"]
    if benchmark.get("status") == "pass":
        for field in (
            "primary_benchmark_id",
            "match_rationale",
            "comparator_definition_path",
        ):
            if not str(benchmark.get(field) or "").strip():
                errors.append(f"benchmark_missing:{field}")

    universe = sections["universe"]
    if universe.get("status") == "pass":
        for field in (
            "research_universe_profile",
            "intended_execution_universe_profile",
            "manifest_path",
        ):
            if not str(universe.get(field) or "").strip():
                errors.append(f"universe_missing:{field}")
        if universe.get("point_in_time_membership") is not True:
            errors.append("universe_point_in_time_membership_must_be_true")
        alignment = str(universe.get("alignment_status") or "").strip()
        if alignment not in {"matched", "mismatch_disclosed"}:
            errors.append("universe_alignment_status_invalid")
        if alignment == "mismatch_disclosed" and not str(
            universe.get("mismatch_reason") or ""
        ).strip():
            errors.append("universe_mismatch_reason_required")

    multiplicity = sections["multiplicity"]
    if multiplicity.get("status") == "pass":
        for field in ("adaptive_lineage_path", "within_family_control"):
            if not str(multiplicity.get(field) or "").strip():
                errors.append(f"multiplicity_missing:{field}")
        inventory_complete = multiplicity.get("historical_trial_inventory_complete")
        global_claimed = multiplicity.get("global_multiplicity_control_claimed")
        if not isinstance(inventory_complete, bool):
            errors.append("multiplicity_inventory_complete_must_be_boolean")
        if not isinstance(global_claimed, bool):
            errors.append("multiplicity_global_claimed_must_be_boolean")
        if global_claimed is True and inventory_complete is not True:
            errors.append("global_multiplicity_claim_requires_complete_inventory")
        if inventory_complete is False:
            warnings.append("historical_trial_inventory_incomplete_lower_bound_only")

    no_future = sections["no_future"]
    if no_future.get("status") == "pass" and not str(
        no_future.get("audit_path") or ""
    ).strip():
        errors.append("no_future_audit_path_required")

    temporal = sections["temporal_stability"]
    if temporal.get("status") == "pass":
        period_kinds = temporal.get("period_kinds")
        artifact_paths = temporal.get("artifact_paths")
        if not isinstance(period_kinds, list) or not period_kinds:
            errors.append("temporal_period_kinds_required")
        if not isinstance(artifact_paths, list) or not artifact_paths:
            errors.append("temporal_artifact_paths_required")

    attribution = sections["attribution"]
    if attribution.get("status") == "pass":
        dimensions = attribution.get("dimensions")
        if not isinstance(dimensions, Mapping):
            errors.append("attribution_dimensions_required")
        else:
            missing_dimensions = sorted(_ATTRIBUTION_DIMENSIONS - set(dimensions))
            for dimension in missing_dimensions:
                errors.append(f"attribution_dimension_missing:{dimension}")
            not_applicable_reasons = attribution.get("not_applicable_reasons")
            for dimension in sorted(_ATTRIBUTION_DIMENSIONS.intersection(dimensions)):
                dimension_status = str(dimensions.get(dimension) or "").strip()
                if dimension_status not in {"pass", "not_applicable"}:
                    errors.append(f"attribution_dimension_invalid:{dimension}")
                if dimension_status == "not_applicable" and (
                    not isinstance(not_applicable_reasons, Mapping)
                    or not str(not_applicable_reasons.get(dimension) or "").strip()
                ):
                    errors.append(
                        f"attribution_not_applicable_reason_required:{dimension}"
                    )
        artifact_paths = attribution.get("artifact_paths")
        if not isinstance(artifact_paths, list) or not artifact_paths:
            errors.append("attribution_artifact_paths_required")

    execution = sections["execution"]
    if execution.get("status") == "pass":
        for field in (
            "cost_audit_path",
            "fill_audit_path",
            "capacity_audit_path",
        ):
            if not str(execution.get(field) or "").strip():
                errors.append(f"execution_missing:{field}")

    promotion_evidence_complete = bool(
        not errors
        and not blocking_gaps
        and stage in {"confirmation", "promotion_audit"}
        and all(
            str(sections[name].get("status") or "") == "pass"
            for name in required_sections
        )
    )
    return {
        "pass": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "blocking_gaps": sorted(set(blocking_gaps)),
        "stage": stage,
        "promotion_evidence_complete": promotion_evidence_complete,
        "contract": {
            "absolute_drawdown_is_relative_drawdown": False,
            "iid_t_stat_is_universal_multiplicity_control": False,
            "undisclosed_attribution_or_execution_gap_may_promote": False,
        },
    }


def assert_research_evidence_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    audit = audit_research_evidence_manifest(payload)
    if not audit["pass"]:
        raise ValueError(f"research evidence manifest guard failed: {audit['errors']}")
    return audit
