# Resumable research execution

`python checkpoint_demo.py` runs an offline walkthrough of the actual kernel: resume a complete unit, refuse reuse under changed input identity, reject a tampered artifact and keep not-yet-observable labels out of fit readiness. The example uses synthetic bytes, not trading results. The runtime modules have been synchronized with the current source implementation.

The runtime is extracted from the same research platform as the point-in-time validation modules. Its job is to preserve execution identity and completed computation across interruption, while keeping scientific acceptance separate from process success.

`qplatform/research_runtime.py` runs a manifest-defined dependency graph. Each stage binds input and code hashes, finite memory/time budgets and required artifacts. Failed branches can block their dependants while independent branches continue. A partial graph remains partial even when every runnable branch exits successfully.

`research_checkpoints.py` publishes complete adapter-defined units under a work-binding hash. A restart verifies artifact hashes before reuse. Changed output, a different binding or incomplete units cannot silently become valid checkpoints. Existing adapters must explicitly use this interface.

`research_process.py` provides owned child-process handling, including Windows Job Objects. `research_readiness.py` checks concrete label availability, fit records and account-continuity artifacts for the isolated research workflow. It does not prove a strategy useful or give execution authority.

```sh
python -B -m pytest -q tests/test_research_runtime.py tests/test_research_runtime_recovery.py
```

The exported tests and worker fixture are the original implementation tests, not real investment runs. They use temporary workspaces and synthetic evidence; no broker, private data, model service or credentials are required. The runtime executes trusted commands and is not a filesystem/network sandbox. Windows-specific sharing and process tests may be skipped on other platforms.

The runtime-only command handler and parser are extracted into `scripts/qpt.py`; the original CLI's broader platform integrations are not exported. Run `python scripts/qpt.py research-runtime --help`, or use the tested Python `run_manifest`, `runtime_status`, `request_stop` and `CheckpointStore` interfaces. License terms for the separately installed psutil dependency remain with its authors.
