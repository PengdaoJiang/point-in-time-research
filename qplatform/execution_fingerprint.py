"""Static local dependency closure and numerical environment for new artifacts."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .research_readiness import file_digest, read_json

PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib",
            "statsmodels", "lightgbm", "psutil")


def numerical_environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "machine": platform.machine(), "packages": versions}


def fingerprint(root: Path, entries: list[Path], *, max_files=1000):
    root = root.resolve()
    entry_paths = [(root / p).resolve() for p in entries]
    platform_root = Path(__file__).resolve().parents[1]
    code_roots = {root, platform_root, *(p.parent for p in entry_paths)}
    pending = list(entry_paths)
    files = {}
    while pending:
        path = pending.pop().resolve()
        if not any(path.is_relative_to(base) for base in code_roots):
            raise ValueError("fingerprint_code_outside_declared_roots")
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        if name in files:
            continue
        if len(files) >= max_files:
            raise ValueError("fingerprint_dependency_budget_exceeded")
        payload = path.read_bytes()
        files[name] = hashlib.sha256(payload).hexdigest()
        if path.suffix != ".py":
            continue
        tree = ast.parse(payload.decode("utf-8-sig"), filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [(base, a.name) for base in (root, platform_root, path.parent) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = path.parent
                if node.level:
                    for _ in range(node.level - 1):
                        base = base.parent
                    modules = [(base, node.module or "")]
                else:
                    modules = [(base, node.module or "") for base in (root, platform_root, path.parent)]
                modules += [(base, (module + "." if module else "") + a.name)
                            for base, module in list(modules) for a in node.names if a.name != "*"]
            for base, module in modules:
                target = base.joinpath(*module.split(".")) if module else base
                if not any(target.is_relative_to(base) for base in code_roots):
                    continue
                for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                    if candidate.is_file():
                        pending.append(candidate)
                for ancestor in target.parents:
                    if ancestor in code_roots or not any(ancestor.is_relative_to(base) for base in code_roots):
                        break
                    init = ancestor / "__init__.py"
                    if init.is_file():
                        pending.append(init)
    return {"schema": "execution_fingerprint_v1",
            "entries": sorted(p.relative_to(root).as_posix() if p.is_relative_to(root) else p.as_posix()
                              for p in entry_paths),
            "code": dict(sorted(files.items())), "environment": numerical_environment(),
            "scope": "static local imports plus declared code; dynamic plugins must be explicit bindings"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declared_code(manifest: dict) -> list[Path]:
    return [Path(ref["path"]) for ref in manifest["bindings"].values()
            if ref["role"] == "entrypoint" or
            (ref["role"] == "science" and Path(ref["path"]).suffix == ".py")]


def verify_manifest_environment(root: Path, manifest: dict, *, required=False):
    key = manifest.get("execution_fingerprint_binding")
    if not key:
        if required:
            raise ValueError("new_research_requires_execution_fingerprint; use research-data prepare")
        return {"status": "legacy_unbound_not_rewritten"}
    ref = manifest["bindings"][key]
    path = (root / ref["path"]).resolve()
    if ref["role"] != "contract" or file_digest(path) != ref["sha256"]:
        raise ValueError("execution_fingerprint_binding_changed")
    expected = read_json(path)
    entries = {(root / p).resolve() for p in expected["entries"]}
    if not {(root / p).resolve() for p in declared_code(manifest)}.issubset(entries):
        raise ValueError("execution_fingerprint_missing_declared_code")
    if fingerprint(root, [Path(p) for p in expected["entries"]]) != expected:
        raise ValueError("execution_dependency_or_environment_changed")
    if any(key not in stage["uses"] for stage in manifest["stages"]):
        raise ValueError("execution_fingerprint_not_bound_to_all_stages")
    return {"status": "pass", "sha256": ref["sha256"]}
