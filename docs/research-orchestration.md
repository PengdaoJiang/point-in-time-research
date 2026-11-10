# Organizing models and computation around research

Long-running research contains different kinds of work: choosing a question, performing deterministic computation, interpreting an event and maintaining the system. The source architecture assigns them different execution paths.

| Responsibility | Work | Handoff |
| --- | --- | --- |
| Research task | Define hypotheses, frozen inputs, comparisons and validity criteria; interpret results | Submit a bounded computation and its required evidence |
| Model-free runner | Schedule dependencies, execute units, checkpoint complete work and package outputs | Emit a terminal or exceptional event with execution identity |
| Bounded model observer | Inspect the selected event and compact result summary when interpretation is useful | Deliver findings or remaining questions to the research task |
| Engineering maintenance | Diagnose repeated runtime/interface failures and improve the execution path | Re-admit the changed environment before compatible work resumes |

This allocation keeps long numerical work in ordinary programs. Model attention is used where a question, judgment or exception needs it; the architecture does not require a model to stay busy polling a running calculation.

## A reproducible comparison

A research task selects a versioned input snapshot, fixes a rolling evaluation schedule and records the code/environment fingerprint. The runner executes complete units and can resume compatible units after interruption. At completion, the result bundle and event retain the run identity; notification delivery and acknowledgment remain separate facts.

The researcher then checks as-of data availability, future-information leakage and temporal dependence before interpreting the comparison. A successful run or reused checkpoint establishes execution progress, not the validity of a hypothesis. If a dependency changes, reuse is decided from the input/code/environment bindings rather than a reassuring old status.

## Code route

- [Data and event runtime](data-and-event-runtime.md): immutable snapshots, selected reads, fingerprints, bundles, dispatch and acknowledgment.
- [Resumable runtime](resumable-runtime.md): dependency scheduling, complete-unit checkpoints and isolated failures.
- [Research cases](case-studies.md): temporal validity and statistical methods from the time-series source application.

The public CLI and synthetic demonstrations run independently of a model service or proprietary data feed. Live model observers require a configured adapter. The source research is time-series work; the extracted runner and event mechanisms provide reusable experiment infrastructure without claiming a deployed scientific discovery system.
