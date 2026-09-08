from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qplatform.research_guard import (  # noqa: E402
    assert_adaptive_research_lineage,
    assert_execution_observability_contract,
    assert_origin_outcome_returns,
    assert_research_account_isolation,
    assert_research_evidence_manifest,
    audit_adaptive_research_lineage,
    audit_execution_observability_contract,
    audit_origin_outcome_returns,
    audit_research_account_isolation,
    audit_research_evidence_manifest,
    allowed_research_feature_columns,
    assert_research_feature_columns,
    audit_research_feature_columns,
    cross_section_rank_score,
)


def main() -> int:
    requested = [
        "ret_5d",
        "amount",
        "entry_open",
        "exit_price_latency_2m",
        "label_ret",
        "future_return",
        "target_weight",
    ]
    audit = audit_research_feature_columns(requested)
    assert audit["pass"] is False, audit
    assert audit["forbidden_features"] == [
        "entry_open",
        "exit_price_latency_2m",
        "future_return",
        "label_ret",
        "target_weight",
    ], audit
    assert allowed_research_feature_columns(requested) == ["amount", "ret_5d"]

    try:
        assert_research_feature_columns(requested)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden feature set was accepted")
    assert assert_research_feature_columns(["ret_5d", "amount"]) == [
        "ret_5d",
        "amount",
    ]
    account_feature_audit = audit_research_feature_columns(
        ["ret_5d", "cash_available_cny", "project_controllable_notional_cny"]
    )
    assert account_feature_audit["pass"] is False
    assert account_feature_audit["forbidden_features"] == [
        "cash_available_cny",
        "project_controllable_notional_cny",
    ]

    isolated_account = {
        "mode": "isolated_synthetic_fixed_capital",
        "initial_cash_cny": 150000,
        "initial_inventory_mode": "flat",
        "real_account_holdings_or_positions_used": False,
        "real_account_cash_margin_or_financing_used": False,
        "real_account_risk_limits_used": False,
        "real_account_transactions_used_as_training_rows_labels_or_weights": False,
        "real_account_state_used_for_candidate_selection_or_rejection": False,
        "synthetic_strategy_occupancy_enforced": True,
        "synthetic_shared_cash_ledger_enforced": True,
        "post_research_execution_overlay_separated": True,
        "research_input_paths": [
            "data_lake/gold/profile_specific/full/dataset_daily/LATEST_train.parquet"
        ],
        "research_input_fields": ["ret_5d", "amount"],
    }
    isolation_audit = assert_research_account_isolation(isolated_account)
    assert isolation_audit["pass"] is True
    assert isolation_audit["initial_cash_cny"] == 150000

    live_holdings_input = dict(
        isolated_account,
        research_input_paths=[
            "workspace_out/state/account/current_holdings.csv"
        ],
    )
    live_holdings_audit = audit_research_account_isolation(live_holdings_input)
    assert live_holdings_audit["pass"] is False
    assert "real_account_state_path_used_as_research_input" in live_holdings_audit[
        "errors"
    ]

    live_capacity_input = dict(
        isolated_account,
        research_input_fields=["ret_5d", "available_margin_cny"],
    )
    live_capacity_audit = audit_research_account_isolation(live_capacity_input)
    assert live_capacity_audit["pass"] is False
    assert "real_account_occupancy_field_used_as_research_input" in (
        live_capacity_audit["errors"]
    )

    real_account_seed = dict(
        isolated_account,
        real_account_holdings_or_positions_used=True,
    )
    seed_audit = audit_research_account_isolation(real_account_seed)
    assert seed_audit["pass"] is False
    assert "real_account_holdings_or_positions_used_must_be_false" in seed_audit[
        "errors"
    ]

    frame = pd.DataFrame(
        {
            "date": ["2026-07-27"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    high_score = cross_section_rank_score(
        frame,
        value_col="value",
        higher_is_better=True,
    )
    low_score = cross_section_rank_score(
        frame,
        value_col="value",
        higher_is_better=False,
    )
    assert int(high_score.idxmax()) == 2, high_score
    assert int(low_score.idxmax()) == 0, low_score

    outcome_surface = pd.DataFrame(
        {
            "action": ["CASH", "BUY", "BUY"],
            "resolved": [True, True, False],
            "net_return": [0.0, 0.012, float("nan")],
        }
    )
    outcome_audit = assert_origin_outcome_returns(
        outcome_surface,
        return_col="net_return",
        resolved_col="resolved",
        action_col="action",
    )
    assert outcome_audit["pass"] is True
    assert outcome_audit["resolved_cash_rows"] == 1

    bad_cash = outcome_surface.copy()
    bad_cash.loc[0, "net_return"] = float("nan")
    bad_cash_audit = audit_origin_outcome_returns(
        bad_cash,
        return_col="net_return",
        resolved_col="resolved",
        action_col="action",
    )
    assert bad_cash_audit["pass"] is False
    assert "resolved_cash_return_must_be_explicit_zero" in bad_cash_audit["errors"]

    zero_filled_unresolved = outcome_surface.copy()
    zero_filled_unresolved.loc[2, "net_return"] = 0.0
    unresolved_audit = audit_origin_outcome_returns(
        zero_filled_unresolved,
        return_col="net_return",
        resolved_col="resolved",
        action_col="action",
    )
    assert unresolved_audit["pass"] is False
    assert "unresolved_outcome_return_must_remain_missing" in unresolved_audit[
        "errors"
    ]

    lineage = {
        "update_class": "new_formula_or_structure_discovery",
        "trigger_class": "explicit_research_program",
        "formula_version": "market_ecology_stateful_profit_search_v1",
        "formula_or_structure_changed": True,
        "specification_frozen_before_evaluation": True,
        "realized_loss_direct_trigger": False,
        "runtime_mutation_allowed": False,
        "parent_run_ids": ["short_horizon_market_ecology_review_v1"],
        "outcome_date_min": "2026-04-20",
        "outcome_date_max": "2026-08-18",
        "reused_parent_outcome_dates": True,
        "reused_outcome_date_count": 118,
        "independent_confirmation_date_count": 0,
        "within_run_declared_trial_count": 1188,
        "known_ancestor_trial_count_lower_bound": 52,
        "cumulative_known_trial_count_lower_bound": 1240,
        "historical_trial_inventory_complete": False,
        "evidence_class": "adaptive_discovery_reused_outcomes",
    }
    lineage_audit = assert_adaptive_research_lineage(lineage)
    assert lineage_audit["pass"] is True
    assert lineage_audit["fresh_independent_evidence"] is False
    assert lineage_audit["global_multiplicity_control_claim_allowed"] is False
    assert "reused_outcomes_are_not_fresh_independent_evidence" in lineage_audit[
        "warnings"
    ]

    false_fresh = dict(lineage, evidence_class="fresh_independent_confirmation")
    false_fresh_audit = audit_adaptive_research_lineage(false_fresh)
    assert false_fresh_audit["pass"] is False
    assert "reused_outcomes_cannot_be_fresh_independent_evidence" in (
        false_fresh_audit["errors"]
    )

    loss_repair = dict(lineage, realized_loss_direct_trigger=True)
    loss_repair_audit = audit_adaptive_research_lineage(loss_repair)
    assert loss_repair_audit["pass"] is False
    assert "realized_loss_must_not_directly_trigger_update" in loss_repair_audit[
        "errors"
    ]

    scheduled_refit = dict(
        lineage,
        update_class="scheduled_parameter_refit",
        trigger_class="data_advance",
        formula_or_structure_changed=False,
        parent_run_ids=[],
        reused_parent_outcome_dates=False,
        reused_outcome_date_count=0,
        within_run_declared_trial_count=1,
        known_ancestor_trial_count_lower_bound=0,
        cumulative_known_trial_count_lower_bound=1,
        historical_trial_inventory_complete=True,
        evidence_class="scheduled_parameter_refit_snapshot",
    )
    assert assert_adaptive_research_lineage(scheduled_refit)["pass"] is True

    evidence_manifest = {
        "stage": "confirmation",
        "return_drawdown": {
            "status": "pass",
            "absolute_equity_curve_path": "research/account_equity.csv",
            "absolute_equity_max_drawdown": -0.12,
            "active_return_equity_curve_path": "research/active_equity.csv",
            "active_return_equity_max_drawdown": -0.08,
            "strategy_to_benchmark_ratio_curve_path": "research/relative_equity.csv",
            "strategy_to_benchmark_ratio_max_drawdown": -0.09,
        },
        "benchmark": {
            "status": "pass",
            "primary_benchmark_id": "equal_weight_point_in_time_opportunity_set",
            "match_rationale": "removes stock-selection decisions while matching the opportunity set and clock",
            "comparator_definition_path": "research/benchmark_manifest.json",
        },
        "universe": {
            "status": "pass",
            "research_universe_profile": "broker_operable_full_active_candidate",
            "intended_execution_universe_profile": "broker_operable_full_active_candidate",
            "point_in_time_membership": True,
            "alignment_status": "matched",
            "manifest_path": "research/universe_manifest.json",
        },
        "multiplicity": {
            "status": "pass",
            "adaptive_lineage_path": "research/adaptive_research_lineage.json",
            "within_family_control": "date_block_bootstrap_max_t_and_bh_fdr",
            "historical_trial_inventory_complete": False,
            "global_multiplicity_control_claimed": False,
        },
        "no_future": {
            "status": "pass",
            "audit_path": "research/no_future_audit.json",
        },
        "temporal_stability": {
            "status": "pass",
            "period_kinds": ["calendar_month", "market_regime", "start_vintage"],
            "artifact_paths": ["research/temporal_stability.csv"],
        },
        "attribution": {
            "status": "pass",
            "dimensions": {
                "market_beta": "pass",
                "industry": "pass",
                "size": "pass",
                "value": "not_applicable",
                "momentum": "pass",
            },
            "not_applicable_reasons": {
                "value": "no PIT value descriptor is available for this intraday route"
            },
            "artifact_paths": ["research/style_industry_attribution.csv"],
        },
        "execution": {
            "status": "pass",
            "cost_audit_path": "research/cost_audit.json",
            "fill_audit_path": "research/fill_audit.json",
            "capacity_audit_path": "research/capacity_audit.json",
        },
    }
    evidence_audit = assert_research_evidence_manifest(evidence_manifest)
    assert evidence_audit["promotion_evidence_complete"] is True
    assert "historical_trial_inventory_incomplete_lower_bound_only" in evidence_audit[
        "warnings"
    ]

    false_global_claim = dict(evidence_manifest)
    false_global_claim["multiplicity"] = dict(
        evidence_manifest["multiplicity"],
        global_multiplicity_control_claimed=True,
    )
    false_global_audit = audit_research_evidence_manifest(false_global_claim)
    assert false_global_audit["pass"] is False
    assert "global_multiplicity_claim_requires_complete_inventory" in false_global_audit[
        "errors"
    ]

    attribution_gap = dict(evidence_manifest)
    attribution_gap["attribution"] = {
        "status": "not_available",
        "gap_reason": "style exposure panel has not been materialized",
    }
    attribution_gap_audit = assert_research_evidence_manifest(attribution_gap)
    assert attribution_gap_audit["pass"] is True
    assert attribution_gap_audit["promotion_evidence_complete"] is False
    assert attribution_gap_audit["blocking_gaps"] == ["attribution"]

    execution_contract = {
        "entry_clock": "t_plus_1_official_open",
        "exit_clock": "t_plus_2_closing_auction",
        "entry_fill_price_source": "raw_open",
        "exit_fill_price_source": "raw_closing_auction_close_proxy",
        "price_limit_policy": "queue_conservative",
        "fill_inputs_available_by_execution_time": True,
        "post_execution_extrema_used_for_fill": False,
        "later_unseal_backfills_earlier_fill": False,
        "upper_limit_buy_without_queue_evidence": "no_fill_cash",
        "lower_limit_sell_without_queue_evidence": "blocked_exit_retry",
        "blocked_exit_retries_at_same_declared_clock": True,
        "right_censored_plans_omitted_not_zero_filled": True,
        "closing_auction_order_predeclared_if_used": True,
        "synthetic_invariance_assertions": {
            "open_at_upper_later_low_does_not_fill": True,
            "close_at_lower_earlier_high_does_not_fill": True,
            "open_exit_at_lower_later_high_does_not_fill": True,
        },
    }
    assert assert_execution_observability_contract(execution_contract)["pass"] is True

    later_low_fill = dict(
        execution_contract,
        post_execution_extrema_used_for_fill=True,
        later_unseal_backfills_earlier_fill=True,
    )
    later_low_audit = audit_execution_observability_contract(later_low_fill)
    assert later_low_audit["pass"] is False
    assert "post_execution_high_low_must_not_determine_fill" in later_low_audit[
        "errors"
    ]
    assert "later_unseal_must_not_backfill_earlier_fill" in later_low_audit[
        "errors"
    ]

    close_not_predeclared = dict(
        execution_contract,
        closing_auction_order_predeclared_if_used=False,
    )
    close_audit = audit_execution_observability_contract(close_not_predeclared)
    assert close_audit["pass"] is False
    assert "closing_auction_order_must_be_predeclared" in close_audit["errors"]

    print("PASS")
    print(
        "research leakage, account-isolation, outcome, execution-observability, "
        "rank, adaptive-lineage, and evidence-manifest guards: ok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
