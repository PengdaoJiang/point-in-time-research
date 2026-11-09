"""Admission and explicit preparation for NEW research, never a legacy rewrite."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from .control_io import atomic_json
from .gc_safety import contained
from .research_readiness import file_digest, read_json

SCHEMA = 'research_storage_contract_v1'
TABLE_SUFFIXES = {'.parquet', '.csv', '.feather', '.arrow', '.ipc'}


def validate_storage(workspace: Path, manifest: dict):
    from .research_batch import validate_plan
    from .research_store import ResearchTable

    workspace = workspace.resolve()
    policy = manifest.get('storage_contract', {})
    if policy.get('schema') != SCHEMA:
        raise ValueError('new_research_requires_shared_storage_contract; use research-data prepare')
    sources = {k: v for k, v in manifest['bindings'].items() if v['role'] == 'source'}
    declarations = policy.get('sources', {})
    if set(declarations) != set(sources):
        raise ValueError('storage_source_declarations_must_cover_exact_bound_sources')
    store = workspace / 'data_lake/research_store/v1'
    for key, ref in sources.items():
        path = (workspace / ref['path']).resolve()
        if file_digest(path) != ref['sha256']:
            raise ValueError(f'storage_source_binding_mismatch:{key}')
        declaration = declarations[key]
        if ref.get('kind') == 'research_snapshot':
            if declaration != {'mode': 'shared_snapshot'} or path.parent != store / 'snapshots':
                raise ValueError(f'source_not_in_shared_store:{key}')
            table = ResearchTable(path)
            table.verify_all()
        elif (declaration.get('mode') != 'immutable_native'
              or not isinstance(declaration.get('reason'), str) or not declaration['reason'].strip()
              or path.suffix.lower() in TABLE_SUFFIXES):
            raise ValueError(f'tabular_sources_require_shared_snapshot:{key}')
    plans = set(policy.get('batch_plans', []))
    used_plans = set()
    for stage in manifest['stages']:
        if not stage.get('parallel_batch'):
            continue
        candidates = plans.intersection(stage['uses'])
        if len(candidates) != 1:
            raise ValueError('parallel_stage_requires_one_bound_v2_batch_plan')
        key = candidates.pop()
        ref = manifest['bindings'][key]
        path = (workspace / ref['path']).resolve()
        if ref['role'] != 'science' or file_digest(path) != ref['sha256']:
            raise ValueError('batch_storage_plan_binding_mismatch')
        # Bind the plan actually named by the stage, not an unused green decoy.
        args = stage['command']
        if args.count('--plan') != 1 or args.index('--plan') + 1 >= len(args):
            raise ValueError('batch_storage_plan_argument_missing')
        if (workspace / args[args.index('--plan') + 1]).resolve() != path:
            raise ValueError('batch_storage_plan_argument_mismatch')
        plan = validate_plan(read_json(path), manifest['bindings'])
        if plan['schema'] != 'research_parallel_batch_v2' or not plan.get('bundle_results') or not plan.get('discard_sealed_work_outputs'):
            raise ValueError('new_batch_requires_packed_results_and_sealed_scratch_retirement')
        used_plans.add(key)
    if used_plans != plans:
        raise ValueError('storage_batch_plan_declarations_mismatch')
    return {'status': 'pass', 'sources': len(sources), 'batch_plans': len(plans)}


def prepare_manifest(workspace: Path, draft: Path, declarations: Path, output: Path):
    from .research_runtime import validate_manifest, runtime_status
    from .research_store import freeze_table

    workspace = workspace.resolve()
    output = contained(workspace, output)
    manifest = copy.deepcopy(read_json(draft))
    validate_manifest(manifest)
    from .research_readiness import validate_new_pilot_admission
    validate_new_pilot_admission(manifest)
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', key) for key in manifest['bindings']):
        raise ValueError('storage_binding_names_must_be_safe_identifiers')
    run_root = workspace / 'workspace_out/runs' / manifest['run_id']
    if not output.is_relative_to(run_root) or output.exists():
        raise ValueError('prepared_manifest_requires_new_file_inside_its_run')
    if runtime_status(workspace, manifest['run_id'])['status'] != 'not_started':
        raise ValueError('cannot_prepare_existing_runtime_or_frozen_run')
    specs = read_json(declarations)
    source_ids = {k for k, ref in manifest['bindings'].items() if ref['role'] == 'source'}
    if set(specs) != source_ids:
        raise ValueError('declare_each_source_once_without_extra_sources')
    policy = {'schema': SCHEMA, 'sources': {}, 'batch_plans': []}
    changes, replacements = [], {}
    for key in sorted(source_ids):
        ref = manifest['bindings'][key]
        source = (workspace / ref['path']).resolve()
        if file_digest(source) != ref['sha256']:
            raise ValueError(f'draft_source_changed:{key}')
        spec = specs[key]
        if source.suffix.lower() == '.parquet':
            if spec.get('reader') != 'ResearchTable':
                raise ValueError(f'explicit_shared_reader_adaptation_required:{key}')
            frozen = freeze_table(workspace / 'data_lake/research_store/v1', [source], spec['semantics'],
                                  partition_by_date=spec.get('partition_by_date', False),
                                  expected_source_hashes=[ref['sha256']])
            replacements[ref['path']] = frozen['snapshot']
            replacements[str(source)] = frozen['snapshot']
            ref.update(path=frozen['snapshot'], sha256=file_digest(Path(frozen['snapshot'])), kind='research_snapshot')
            policy['sources'][key] = {'mode': 'shared_snapshot'}
            changes.append({'binding': key, 'source': str(source), **frozen})
        elif ref.get('kind') == 'research_snapshot':
            policy['sources'][key] = {'mode': 'shared_snapshot'}
        else:
            policy['sources'][key] = {'mode': 'immutable_native', 'reason': spec.get('reason', '')}
    # New prepared files only. No existing scientific plan or argument is edited.
    for key, ref in manifest['bindings'].items():
        source = (workspace / ref['path']).resolve()
        if ref['role'] != 'science' or source.suffix.lower() != '.json':
            continue
        plan = read_json(source)
        if not isinstance(plan, dict) or plan.get('schema') not in {'research_parallel_batch_v1', 'research_parallel_batch_v2'}:
            continue
        if file_digest(source) != ref['sha256']:
            raise ValueError('draft_batch_plan_changed')
        if plan.get('bundle_results') is False or plan.get('discard_sealed_work_outputs') is False:
            raise ValueError('explicit_loose_storage_conflicts_with_new_storage_contract')
        plan = {**plan, 'schema': 'research_parallel_batch_v2',
                'bundle_results': True, 'discard_sealed_work_outputs': True}
        for job in plan['jobs']:
            job['args'] = [replacements.get(arg, arg) for arg in job['args']]
        target = output.parent / (output.stem + '.plans') / f'{key}.json'
        contained(workspace, target)
        if target.exists():
            raise ValueError('prepared_batch_plan_already_exists')
        atomic_json(target, plan)
        old_path = ref['path']
        ref.update(path=str(target), sha256=file_digest(target))
        for stage in manifest['stages']:
            stage['command'] = [str(target) if arg in {old_path, str(source)} else arg for arg in stage['command']]
        policy['batch_plans'].append(key)
    for stage in manifest['stages']:
        stage['command'] = [replacements.get(arg, arg) for arg in stage['command']]
    manifest['storage_contract'] = policy
    from .execution_fingerprint import declared_code, fingerprint, verify_manifest_environment
    key = "execution_environment"
    if key in manifest["bindings"]:
        raise ValueError("execution_environment_binding_reserved")
    entries = declared_code(manifest)
    if not entries:
        raise ValueError("execution_fingerprint_requires_declared_code")
    environment_path = output.with_name(output.stem + ".environment.json")
    if environment_path.exists():
        raise ValueError("execution_environment_already_exists")
    atomic_json(environment_path, fingerprint(workspace, entries))
    manifest["bindings"][key] = {"path": str(environment_path), "sha256": file_digest(environment_path),
                                "role": "contract"}
    manifest["execution_fingerprint_binding"] = key
    for stage in manifest["stages"]:
        stage["uses"] = [*stage["uses"], key]
    verify_manifest_environment(workspace, manifest, required=True)
    validate_manifest(manifest)
    validate_storage(workspace, manifest)
    atomic_json(output, manifest)
    return {'status': 'prepared_not_started', 'manifest': str(output), 'sha256': file_digest(output),
            'sources': changes, 'runtime_started': False, 'canonical_refresh': False}
