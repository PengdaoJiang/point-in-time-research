"""Parallel frozen experiments inside (never instead of) research-runtime.

No model calls, formula generation, automatic retries or outcome-based pruning.
The runtime owns the whole process tree, hard memory limit, stop and outbox.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .control_io import atomic_json
from .research_checkpoints import CheckpointStore
from .research_readiness import artifact_path, file_digest, read_json
from .research_result_bundle import ResultBundle, build_bundle


def validate_plan(plan, bindings):
    if plan.get("schema") not in {"research_parallel_batch_v1", "research_parallel_batch_v2"}:
        raise ValueError("unknown batch schema")
    plan = dict(plan)
    if plan['schema'] == 'research_parallel_batch_v2':
        plan.setdefault('bundle_results', True)
        plan.setdefault('discard_sealed_work_outputs', True)
    for key in ('bundle_results', 'discard_sealed_work_outputs'):
        if type(plan.get(key, False)) is not bool:
            raise ValueError(f'{key} must be boolean')
    if not isinstance(plan.get("question"), str) or not plan["question"].strip():
        raise ValueError("batch must declare a research question")
    spec = plan.get("research_spec_binding")
    if spec not in bindings or bindings[spec]["role"] != "contract":
        raise ValueError("batch requires a frozen scientific predeclaration")
    for key in ("worker_memory_bytes", "worker_threads"):
        if type(plan.get(key)) is not int or plan[key] <= 0:
            raise ValueError("positive worker resources required")
    known = set()
    if not isinstance(plan.get("jobs"), list) or not plan["jobs"]:
        raise ValueError("empty batch")
    for job in plan["jobs"]:
        name = job["id"]
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name) or name in known:
            raise ValueError("invalid/duplicate experiment id")
        known.add(name)
        if job["entrypoint_binding"] not in bindings or bindings[job["entrypoint_binding"]]["role"] != "entrypoint":
            raise ValueError("worker code must be bound")
        if job.get("role") not in {"main_effect", "interaction", "ablation", "negative_control", "account_replay", "diagnostic"}:
            raise ValueError("declare experiment role")
        if not isinstance(job.get("args"), list) or not all(isinstance(a, str) for a in job["args"]):
            raise ValueError("worker args must be an argv list")
        if not job.get("outputs") or len(set(job["outputs"])) != len(job["outputs"]):
            raise ValueError("unique complete outputs required")
        for output in job["outputs"]:
            artifact_path(Path.cwd(), output)
    return plan


def verify_results(path: Path, checkpoint_root: Path, work_binding: str):
    result = read_json(path)
    if result.get("schema") != "research_batch_results_v1" or result.get("work_binding_id") != work_binding or not result.get("complete"):
        raise ValueError("incomplete/mismatched batch result")
    if not result.get("jobs") or result["total_units"] != len(result["jobs"]):
        raise ValueError("batch coverage mismatch")
    with CheckpointStore(checkpoint_root, work_binding) as store:
        for unit, ref in result["jobs"].items():
            loaded = store.load(unit)
            if loaded is None or ref["receipt_sha256"] != loaded["receipt_sha256"]:
                raise ValueError(f"batch checkpoint mismatch:{unit}")
    if result.get('result_bundle'):
        ref = result['result_bundle']
        with ResultBundle(artifact_path(path.parent, ref['path']), ref['sha256']) as bundle:
            bundle.verify_all()
            if bundle.manifest['units'] != result['total_units']:
                raise ValueError('batch bundle coverage mismatch')
            for unit, job in result['jobs'].items():
                receipt = bundle.unit_receipt(unit, work_binding)
                if receipt['receipt_sha256'] != job['receipt_sha256']:
                    raise ValueError('batch bundle receipt mismatch')
    return result


def _discard_work_outputs(directory: Path, outputs: Path, sealed: dict):
    if not directory.resolve().is_relative_to((outputs / 'work').resolve()):
        raise ValueError('worker cleanup outside owned scratch')
    paths = [(artifact_path(directory, name), sha) for name, sha in sealed['receipt']['artifacts'].items()]
    if any(file_digest(path) != sha for path, sha in paths):
        raise ValueError('worker output changed after checkpoint')
    atomic_json(directory / 'sealed_outputs.json', {
        'receipt_sha256': sealed['receipt_sha256'], 'checkpoint': str(sealed['directory']),
        'outputs': sealed['receipt']['artifacts'], 'policy': 'explicit_discard_sealed_work_outputs',
    })
    for path, _ in paths:
        path.unlink()


def _bundle_records(store, finished):
    for unit in sorted(finished):
        loaded = store.load(unit)
        yield {'unit': unit, 'kind': 'receipt', 'name': 'receipt.json',
               'source': loaded['directory'].parent / 'receipt.json', 'sha256': loaded['receipt_sha256']}
        for name, sha in loaded['receipt']['artifacts'].items():
            yield {'unit': unit, 'name': name, 'source': artifact_path(loaded['directory'], name), 'sha256': sha}


def run_batch(plan_path: Path):
    if os.environ.get("QPT_PARALLEL_BATCH") != "1":
        raise ValueError("parallel batch may only run under research-runtime")
    binding = read_json(Path(os.environ["QPT_RUNTIME_BINDING_FILE"]))
    bindings = binding["inputs"]
    plan_path = plan_path.resolve()
    if not any(Path(ref["path"]).resolve() == plan_path and file_digest(plan_path) == ref["sha256"] for ref in bindings.values()):
        raise ValueError("batch plan is not a verified stage input")
    plan = validate_plan(read_json(plan_path), bindings)
    # Recheck before launching; all child entrypoints must have immutable bindings.
    for ref in bindings.values():
        if file_digest(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("batch input changed after admission")
    outputs = Path(os.environ["QPT_OUTPUT_DIR"])
    stop = Path(os.environ["QPT_STOP_FILE"])
    store = CheckpointStore.from_environment()
    cap = int(os.environ["QPT_STAGE_MEMORY_BYTES"])
    floor = int(os.environ["QPT_STAGE_AVAILABLE_FLOOR_BYTES"])
    overhead = 128 * 1024**2
    cpu_slots = max(1, (os.cpu_count() or 1) // plan["worker_threads"])
    memory_slots = (cap - overhead) // plan["worker_memory_bytes"]
    if memory_slots < 1:
        raise ValueError("stage budget cannot admit one worker plus coordinator")
    slots = min(cpu_slots, memory_slots)
    requested = os.environ.get("QPT_RESEARCH_WORKERS")
    if requested:
        if int(requested) < 1:
            raise ValueError("QPT_RESEARCH_WORKERS must be positive")
        slots = min(slots, int(requested))
    pending, active, finished, failed = list(plan["jobs"]), {}, {}, {}
    start = time.monotonic()
    peak_workers = 0

    def publish():
        atomic_json(outputs / "progress.json", {
            "completed_units": len(finished), "failed_units": len(failed), "total_units": len(plan["jobs"]),
            "active_units": sorted(active), "pending_units": len(pending), "worker_slots": slots,
            "elapsed_s": time.monotonic() - start, "stop_requested": stop.exists(),
        })

    try:
        while pending or active:
            if stop.exists() and not active:
                break
            while pending and len(active) < slots and not stop.exists():
                job = pending[0]
                cached = store.load(job["id"])
                if cached:
                    finished[job["id"]] = {"receipt_sha256": cached["receipt_sha256"], "reused": True,
                                           "role": job["role"]}
                    pending.pop(0)
                    continue
                if psutil.virtual_memory().available < floor + plan["worker_memory_bytes"]:
                    break  # Existing parent wall deadline and memory policy remain authoritative.
                pending.pop(0)
                directory = outputs / "work" / job["id"]
                directory.mkdir(parents=True)
                ref = bindings[job["entrypoint_binding"]]
                entrypoint = Path(ref["path"]).resolve()
                command = [sys.executable, str(entrypoint)] if entrypoint.suffix == ".py" else [str(entrypoint)]
                for arg in job["args"]:
                    arg = arg.replace("{output_dir}", str(directory))
                    for key, item in bindings.items():
                        arg = arg.replace("{input:" + key + "}", str(Path(item["path"]).resolve()))
                    command.append(arg)
                environment = os.environ.copy()
                environment["QPT_OUTPUT_DIR"] = str(directory)
                # Workers cannot seal into or collide with the coordinator's unit namespace.
                for key in ("QPT_CHECKPOINT_DIR", "QPT_WORK_BINDING_ID", "QPT_PARALLEL_BATCH"):
                    environment.pop(key, None)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
                    environment[key] = str(plan["worker_threads"])
                out = (directory / "stdout.log").open("wb")
                err = (directory / "stderr.log").open("wb")
                try:
                    process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                               stdout=out, stderr=err, shell=False,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except BaseException:
                    out.close()
                    err.close()
                    raise
                active[job["id"]] = (process, job, directory, out, err)
                peak_workers = max(peak_workers, len(active))
            for name, (process, job, directory, out, err) in list(active.items()):
                if process.poll() is None:
                    continue
                out.close()
                err.close()
                del active[name]
                if process.returncode:
                    failed[name] = f"exit_code:{process.returncode}"
                    continue
                try:
                    sealed = store.seal(name, directory, job["outputs"])
                    if plan.get('discard_sealed_work_outputs'):
                        _discard_work_outputs(directory, outputs, sealed)
                    finished[name] = {"receipt_sha256": sealed["receipt_sha256"], "reused": False, "role": job["role"]}
                except (OSError, ValueError) as exc:
                    failed[name] = str(exc)
            publish()
            if pending or active:
                time.sleep(.1)
        result = {"schema": "research_batch_results_v1", "work_binding_id": store.binding,
                  "complete": len(finished) == len(plan["jobs"]), "total_units": len(plan["jobs"]),
                  "jobs": finished, "failed": failed, "pending": [j["id"] for j in pending],
                  "peak_workers": peak_workers, "worker_slots": slots, "elapsed_s": time.monotonic() - start}
        if result['complete'] and plan.get('bundle_results'):
            previous = store.packed_bundle()
            if previous and previous[1]['receipts'] == {k: v['receipt_sha256'] for k, v in finished.items()}:
                # A resumed attempt can reuse a complete package without
                # unpacking or rerunning scientific units.
                previous[0].verify_all()
                destination = outputs / 'result_bundle'
                shutil.copytree(previous[0].root, destination)
                packed = {**previous[0].manifest, 'manifest': str(destination / 'manifest.json'),
                          'sha256': previous[1]['sha256'], 'files': len(previous[0].manifest['parts']) + 2,
                          'stored_bytes': sum(p.stat().st_size for p in destination.iterdir())}
            else:
                packed = build_bundle(outputs / 'result_bundle', _bundle_records(store, finished))
            store.register_bundle(Path(packed['manifest']), packed['sha256'])
            result['result_bundle'] = {'path': 'result_bundle/manifest.json', 'sha256': packed['sha256'],
                                       'files': packed['files'], 'stored_bytes': packed['stored_bytes'],
                                       'logical_bytes': packed['logical_bytes']}
        atomic_json(outputs / "batch_results.json", result)
        return result
    finally:
        store.close()
        # On coordinator failure the runtime closes its job/process group, including descendants.
        for process, _, _, out, err in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            out.close()
            err.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run_batch(args.plan)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
