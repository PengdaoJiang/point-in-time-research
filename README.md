# Reproducible Experiment Runtime

Research orchestration, versioned data and reproducible computation for long-running experiments. Extracted from a maintained time-series platform, the architecture assigns question design and result judgment to the research task, long numerical work to a model-free runner, and bounded event interpretation to an optional model observer.

The [orchestration guide](docs/research-orchestration.md) explains that division of work and its connection to experiment design, checkpoint reuse and event handoff. Temporal validation separately determines which conclusions the completed computation can support.

## Choose a reading route

| Layer | What it does | Guide |
| --- | --- | --- |
| Execution | Dependency scheduling, complete compute-unit checkpoints, process supervision, memory pressure and isolated failure branches | [Resumable runtime](docs/resumable-runtime.md) |
| Data and handoff | Immutable Parquet snapshots, content reuse, selected-column reads, environment fingerprints, result bundles and acknowledged terminal events | [Data and event runtime](docs/data-and-event-runtime.md) |
| Experimental validity | As-of membership, future-information checks, rolling windows, dependence statistics and temporal diagnostics | [Research cases](docs/case-studies.md) |

## Run offline

Python 3.10+:

```sh
python -m pip install -r requirements.txt
python -B -m pytest -q
python -B demo.py --out demo-output
python -B data_plane_demo.py
python -B scripts/qpt.py research-dispatch --help
```

Examples use synthetic inputs and write their own outputs. `demo-output/case.json`, `coverage.csv` and `validation/summary.md` connect results to the source implementation. No market-data account or model service is needed.

## Important distinctions

- A reusable checkpoint binds input content, code and numerical dependencies to a complete compute unit. Corrupt or mismatched prior work is refused.
- A calculation can finish before its notification is delivered or acknowledged. The event runtime preserves those separate states.
- A valid input snapshot does not establish an unbiased research conclusion. The temporal checks and statistical methods remain separate from execution success.
- Data access can select partitions and columns without loading an entire historical carrier into each worker.

The original application is market time-series research. Its account-facing integrations and proprietary data are excluded. Public demonstrations are synthetic and make no trading-performance claim. The reusable execution layer is independently inspectable alongside that domain-specific validation code.

Module identities and extraction adaptations are recorded in [source provenance](docs/source-provenance.md). The repository retains the actual implementation and original offline regression cases.

## License

[MIT](LICENSE). NumPy, pandas, PyArrow and pytest are installed separately under their own licenses.
