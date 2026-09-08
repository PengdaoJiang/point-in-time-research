# Three engineering cases

## Membership observed later must not repair an earlier decision

The original research platform needed to distinguish historical membership coverage from a present-day operable universe. `resolve_asof_membership` walks ordered snapshots and never chooses a date after the decision. A missing earlier snapshot stays uncovered; an old one is rejected rather than silently carried forever.

The public fixture makes the change visible: one symbol disappears, another arrives, and decisions before the first snapshot and after the age limit are retained as audit rows. The original negative tests remain in `tests/test_point_in_time_universe.py`.

## Overlapping windows are not independent trials

`build_sequential_validation` computes every usable origin for each requested horizon. The same dates are reused, so the report keeps this as current-history stability evidence. Date-level mean inference uses HAC and moving blocks; overlap sets a minimum lag/block length. Degenerate block evidence does not yield a spurious confidence interval.

The code also examines absolute/squared returns. Weak linear autocorrelation is not proof that returns are independent or unpredictable. These interpretation distinctions are implemented in output fields and exercised in the retained tests.

## A healthy data pipeline is not permission to trade

Research feature and manifest checks keep isolated experiments separate from real-account state and later execution feasibility. Their purpose is to make violations visible and reject inconsistent evidence, not to certify an arbitrary strategy from declarations alone. The public kernel contains no order connector or automatic account action.

The demo is a newly constructed regression case using the same implementation. It is not a replay of an actual account or a historical profit claim.
