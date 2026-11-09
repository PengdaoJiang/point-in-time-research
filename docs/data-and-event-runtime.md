# Shared data and terminal-event delivery

The November extraction separates scientific decisions, deterministic compute,
terminal observation and delivery. A model is not kept alive to poll a long
calculation. The runtime emits a bounded terminal packet, an optional observer
interprets it once, and the owning research context receives a scoped event.

| Layer | Code | Contract |
| --- | --- | --- |
| Shared immutable inputs | `research_store.py` | Exact source hashes/schema/semantics; content-addressed objects; projected reads |
| Compute and recovery | `research_runtime.py`, `research_batch.py`, `research_checkpoints.py` | Bound complete units; dependency-aware resume; failure isolation |
| Environment and storage admission | `execution_fingerprint.py`, `research_storage_policy.py` | Exact code/environment and declared input bindings |
| Result delivery | `research_result_bundle.py` | Verify and restore selected unit artifacts without recomputation |
| Terminal observer | `research_observer.py` | Finite packet, time and memory; no automatic retry of an uncertain attempt |
| Orchestration | `research_dispatch.py` | Same-run identity, pause priority, explicit handoff and continuation |
| Receipt health | `research_delivery.py`, `research_watchdog.py` | Queued, received, acknowledged, overdue and uncertain are distinct |

An immutable input snapshot may be reused by multiple runs. A changed source
produces a different identity; old readers keep their selected version. Dates
are typed and validated before partitioning. Projection and filtering avoid
materializing columns or partitions a worker did not request.

## Offline use

```sh
python -B -m pytest tests -q
python -B data_plane_demo.py
python -B scripts/qpt.py research-runtime --help
python -B scripts/qpt.py research-dispatch --help
```

All regression tests use synthetic temporary inputs, dummy processes and mocked
model/transport calls. The example makes no market-data request and no trade.

The dispatch CLI exposes status/receive/ack/stop and explicit start/handoff.
Start/handoff may invoke the configured Codex CLI and enqueue a message. They
require an explicitly supplied policy; the bundled policy has synthetic UUIDs
and is for inspection/tests. A real deployment must supply its own exact target
identities and qualified route. The source validator intentionally retains its
tested observer alias rather than promising arbitrary provider compatibility.

An acknowledgement proves consumption of the bound event, not approval of a
research hypothesis. The runtime does not promote a strategy, certify causality
or infer profitability. A delayed acknowledgement remains late even after it
arrives; an ambiguous delivery is not silently replayed.

## Extraction record

Fifteen runtime/data modules, eight source regression files and two dummy worker
fixtures were selected from the maintained platform. Account-facing code,
strategy runners, live run state and datasets are outside this extraction.
Three tests whose only subject was the absent strategy consumer were excluded;
the runtime/data tests remain. The public CLI retains the source dispatch
handler and makes the routing policy explicit for start/handoff.

Public portability fixes close Parquet readers on validation exceptions and
declare the empty-fixture date type across pandas versions. These are recorded
adaptations; the private platform was not changed by this release. Full local
regression, corruption/recovery cases and both operating-system CI jobs are the
validation path for this checkout.
