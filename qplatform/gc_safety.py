"""Conservative retention graph for live control-plane references.

Absence from this graph is not proof that a legacy experiment is disposable.
Unscoped age-based apply is intentionally disabled; retirement requires a
separate, hashed inventory and an explicit reason.
"""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

MANDATORY = (
    "cross_section", "minute_raw", "day_raw", ".git", "configs", "docs",
    "qplatform", "scripts", "strategies", "tasks", "tools", ".agent",
    "workspace_out/ops", "workspace_out/state", "workspace_out/orders",
    "data_lake/state", "data_lake/research_store", "data_lake/market_publications",
)
HISTORICAL = ("workspace_out/.trash", "workspace_out/archive",
              "workspace_out/handoff", "workspace_out/review_packages")


def overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def contained(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path if path.is_absolute() else root / path
    # Check lexical ancestors before resolve(), which would hide a junction.
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("gc_path_escape")
    for member in (path, *path.parents):
        if member == root:
            break
        if member.exists() and is_link(member):
            raise ValueError("gc_reparse_path")
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("gc_path_escape")
    return resolved


def walk_files(directory: Path):
    if not directory.exists():
        return
    if is_link(directory):
        raise ValueError(f"gc_reparse_path:{directory}")
    def fail(error):
        raise error
    for base, dirs, files in os.walk(directory, followlinks=False, onerror=fail):
        parent = Path(base)
        for name in dirs + files:
            if is_link(parent / name):
                raise ValueError(f"gc_reparse_path:{parent / name}")
        for name in files:
            yield parent / name


def retention_graph(root: Path, *, max_files=50000, max_bytes=256 * 1024**2) -> dict:
    root = root.resolve()
    references, errors, documents, seen = {}, [], [], set()
    parsed_references = {}
    runs = root / "workspace_out/runs"
    run_ids = {p.name for p in runs.iterdir() if p.is_dir() and not is_link(p)} if runs.exists() else set()
    queue = []
    # State is explicit live authority. Historical event logs are not parsed.
    for relative in ("configs", "workspace_out/state", "data_lake/state"):
        try:
            for p in walk_files(root / relative):
                if p.suffix.lower() != ".json":
                    continue
                live_name = p.stem.lower()
                if (relative == "configs" or p.parent == root / relative
                        or any(token in live_name for token in ("current", "latest", "state", "manifest", "registry"))):
                    queue.append(p)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    for run_id in sorted(run_ids):
        for surface in ("research_runtime", "research_dispatch"):
            for name in ("current.json", "request.json", "manifest.json"):
                p = runs / run_id / "operations" / surface / name
                if p.exists():
                    queue.append(p)
    used = 0

    def reference(raw, origin, key=""):
        if raw not in run_ids and "workspace_out" not in raw and "data_lake" not in raw:
            return
        if raw in parsed_references:
            for rel in parsed_references[raw]:
                references.setdefault(rel, set()).add(origin)
            return
        parsed_references[raw] = []
        value = raw.replace("\\", "/").strip()
        if value in run_ids:
            value = "workspace_out/runs/" + value
        found = re.search(r"(?:workspace_out|data_lake)/[^\n\r\"<>|]+", value)
        if not found:
            return
        value = found.group(0).rstrip(" `;,)")
        # A structured string may contain prose after a path. Run-root protection
        # remains reliable even when its leaf cannot be parsed as a filename.
        parts = value.split("/")
        if value.rstrip("/") in {"workspace_out", "workspace_out/runs", "data_lake"}:
            return
        if len(parts) >= 3 and parts[:2] == ["workspace_out", "runs"]:
            run_id = parts[2].split()[0] if parts[2].strip() else ""
            if run_id in run_ids:
                run_ref = "workspace_out/runs/" + run_id
                references.setdefault(run_ref, set()).add(origin)
                parsed_references[raw].append(run_ref)
        try:
            target = contained(root, Path(value))
        except (ValueError, OSError):
            errors.append(f"invalid_reference:{origin}:{value}")
            return
        rel = target.relative_to(root).as_posix()
        references.setdefault(rel, set()).add(origin)
        parsed_references[raw].append(rel)
        if any(target.is_relative_to(root / h) for h in HISTORICAL):
            return
        if target.is_file() and target.suffix.lower() == ".json":
            queue.append(target)

    def visit(value, origin, key=""):
        if isinstance(value, dict):
            if value.get('schema') == 'storage_recovery_pin_v1':
                files = value.get('files', [])
                if not isinstance(files, list):
                    raise ValueError('invalid_retained_file_pin')
                for entry in files:
                    if (not isinstance(entry, dict) or not isinstance(entry.get('path'), str)
                            or not re.fullmatch('[0-9a-f]{64}', str(entry.get('sha256', '')))):
                        raise ValueError('invalid_retained_file_pin')
                    target = contained(root, Path(entry['path']))
                    if (not target.is_file()
                            or any(target.is_relative_to(root / h) for h in HISTORICAL)):
                        raise ValueError('retained_file_pin_missing_or_historical')
                    # Pins must also protect files outside the usual data/run roots.
                    references.setdefault(target.relative_to(root).as_posix(), set()).add(origin)
            for k, v in value.items():
                visit(v, origin, k)
        elif isinstance(value, list):
            for v in value:
                visit(v, origin, key)
        elif isinstance(value, str):
            reference(value, origin, key)

    # Canonical prose/code may retain a run without a registry entry. Keep exact
    # identifiers and versioned dataset filenames; do not execute or import code.
    versions = {p.name.lower(): p for p in (root / 'data_lake/gold/dataset_daily').glob('*.parquet')}
    for directory in ('docs', 'qplatform', 'scripts', 'tools', 'strategies', 'tasks'):
        try:
            for path in walk_files(root / directory):
                if path.suffix.lower() not in {'.py', '.md', '.ps1', '.json'}:
                    continue
                used += path.stat().st_size
                if used > max_bytes:
                    raise ValueError('retention_graph_budget_exceeded')
                content = path.read_text(encoding='utf-8-sig')
                origin = path.relative_to(root).as_posix()
                documents.append(origin)
                for token in set(re.findall(r'[A-Za-z0-9][A-Za-z0-9_\-\.]+', content)):
                    if token in run_ids:
                        references.setdefault('workspace_out/runs/' + token, set()).add(origin)
                    if token.lower() in versions:
                        references.setdefault(versions[token.lower()].relative_to(root).as_posix(), set()).add(origin)
        except (OSError, ValueError) as exc:
            errors.append(f'unreadable_canonical_reference:{directory}:{exc}')

    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            path = contained(root, path)
            size = path.stat().st_size
            used += size
            if len(seen) > max_files or used > max_bytes:
                errors.append("retention_graph_budget_exceeded")
                break
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            relative = path.relative_to(root).as_posix()
            documents.append(relative)
            visit(data, relative)
        except (OSError, ValueError) as exc:
            errors.append(f"unreadable_authority:{path}:{exc}")
    return {"schema": "gc_retention_graph_v1", "complete": not errors, "errors": errors,
            "documents": sorted(documents), "bytes_read": used,
            "references": {k: sorted(v) for k, v in sorted(references.items())},
            "scope": "canonical code/doc exact run and dataset references; configs, live state/current/latest/manifest/registry pointers, dispatch manifests and transitive JSON; dated snapshots excluded; legacy absence is not disposal proof"}


def protections(root: Path, graph: dict) -> list[Path]:
    root = root.resolve()
    policy_path = root / "configs/protected_roots.json"
    extra = []
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
        extra = policy.get("protected_roots") if isinstance(policy, dict) else None
        if not isinstance(extra, list) or any(not isinstance(p, str) or not p.strip() for p in extra):
            raise ValueError("invalid_protected_roots_policy")
    # Graph targets were already checked while resolving references. Rewalking
    # tens of thousands of ancestors here multiplies every per-target GC check.
    return [root / p for p in (*MANDATORY, *HISTORICAL, *graph["references"])] + [contained(root, Path(p)) for p in extra]


def is_latest_publication(path: Path) -> bool:
    name = path.name.upper()
    return name.startswith("LATEST") or "LATEST" in re.split(r"[_.-]", name)


def assert_retirable(root: Path, paths: list[Path], graph: dict):
    if not graph["complete"]:
        raise ValueError("incomplete_retention_graph:" + ";".join(graph["errors"][:3]))
    protected = protections(root, graph)
    for path in paths:
        contained(root, path)
        if any(overlap(path, p) for p in protected):
            raise ValueError(f"referenced_or_protected:{path}")
        if is_latest_publication(path):
            raise ValueError("canonical_latest_not_retirable")
        for ancestor in (path, *path.parents):
            if ancestor == root.resolve():
                break
            if ancestor.is_dir() and (ancestor / "PINNED").exists():
                raise ValueError("pinned_evidence_not_retirable")
        if path.is_dir() and any(is_latest_publication(p) for p in walk_files(path)):
            raise ValueError("canonical_latest_not_retirable")
