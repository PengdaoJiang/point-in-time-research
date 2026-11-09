"""Atomic, content-verified units for science adapters using research-runtime.

The adapter chooses a complete scientific unit and its serialization. The
platform only seals and reuses bytes, never partial calculations or economics.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from .research_readiness import artifact_path, file_digest, read_json
from .research_runtime import _atomic, _digest
from .research_result_bundle import ResultBundle


class CheckpointStore:
    def __init__(self, root: Path, work_binding_id: str):
        if not re.fullmatch(r"[0-9a-f]{64}", work_binding_id):
            raise ValueError("checkpoint requires a runtime work binding")
        self.binding = work_binding_id
        self.root = Path(root).resolve() / work_binding_id
        self._bundle = None
        self._bundle_signature = None

    def close(self):
        if self._bundle is not None:
            self._bundle.close()
            self._bundle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def packed_bundle(self):
        path = self.root / 'packed.json'
        if not path.exists():
            return None
        ref = read_json(path)
        if ref.get('schema') != 'research_packed_checkpoints_v1' or ref.get('work_binding_id') != self.binding:
            raise ValueError('packed_checkpoint_binding_mismatch')
        manifest = Path(ref['manifest'])
        members = (path, manifest, manifest.parent / 'index.sqlite')
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in members)
        if self._bundle is None or signature != self._bundle_signature:
            self.close()
            self._bundle = ResultBundle(manifest, ref['sha256'])
            self._bundle_signature = signature
        return self._bundle, ref

    def register_bundle(self, manifest: Path, expected_sha256: str):
        with ResultBundle(manifest, expected_sha256) as bundle:
            bundle.verify_all()
            units = [row[0] for row in bundle.db.execute('SELECT DISTINCT unit FROM artifacts')]
            receipts = {}
            for unit in units:
                packed = bundle.unit_receipt(unit, self.binding)
                loose = self.load(unit)
                if loose is None or loose['receipt_sha256'] != packed['receipt_sha256']:
                    raise ValueError('packed_checkpoint_source_mismatch')
                receipts[unit] = packed['receipt_sha256']
        existing = self.packed_bundle()
        if existing:
            if existing[1]['receipts'] != receipts:
                raise ValueError('packed_checkpoint_immutable_conflict')
            return existing[1]
        ref = {'schema': 'research_packed_checkpoints_v1', 'work_binding_id': self.binding,
               'manifest': str(Path(manifest).resolve()), 'sha256': expected_sha256, 'receipts': receipts}
        _atomic(self.root / 'packed.json', ref)
        return ref

    @classmethod
    def from_environment(cls) -> "CheckpointStore":
        return cls(Path(os.environ["QPT_CHECKPOINT_DIR"]), os.environ["QPT_WORK_BINDING_ID"])

    def _directory(self, unit_id: str) -> Path:
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError("checkpoint unit id must be nonempty")
        return self.root / _digest(unit_id)

    def load(self, unit_id: str) -> dict[str, Any] | None:
        directory = self._directory(unit_id)
        if not directory.exists():
            packed = self.packed_bundle()
            if packed is None or unit_id not in packed[1]['receipts']:
                return None
            bundle, ref = packed
            loaded = bundle.unit_receipt(unit_id, self.binding)
            if loaded['receipt_sha256'] != ref['receipts'][unit_id]:
                raise ValueError('packed_checkpoint_receipt_mismatch')
            # Do not rehydrate every unit merely to count/reuse completed work.
            # Legacy path-based consumers must explicitly materialize before use.
            for name in loaded['receipt']['artifacts']:
                with bundle.open_artifact(unit_id, name):
                    pass
            return {**loaded, 'directory': None, 'packed': ref}
        receipt = read_json(directory / "receipt.json")
        if (receipt.get("schema") != "research_unit_checkpoint_v1"
                or receipt.get("unit_id") != unit_id
                or receipt.get("work_binding_id") != self.binding
                or not receipt.get("artifacts")):
            raise ValueError(f"checkpoint_identity_mismatch:{unit_id}")
        for name, digest in receipt["artifacts"].items():
            if file_digest(artifact_path(directory / "artifacts", name)) != digest:
                raise ValueError(f"checkpoint_artifact_mismatch:{unit_id}:{name}")
        return {"directory": directory / "artifacts", "receipt": receipt,
                "receipt_sha256": file_digest(directory / "receipt.json")}

    def read_bytes(self, unit_id: str, name: str):
        loaded = self.load(unit_id)
        if loaded is None or name not in loaded['receipt']['artifacts']:
            raise KeyError((unit_id, name))
        if loaded['directory'] is not None:
            return artifact_path(loaded['directory'], name).read_bytes()
        return self.packed_bundle()[0].read_bytes(unit_id, name)

    def restore_unit(self, unit_id: str, destination: Path):
        packed = self.packed_bundle()
        if packed is None or unit_id not in packed[1]['receipts']:
            raise ValueError('unit_has_no_verified_result_package')
        self.load(unit_id)
        return packed[0].restore_unit(unit_id, destination, binding=self.binding)

    def seal(self, unit_id: str, source: Path, outputs: list[str]) -> dict[str, Any]:
        if not outputs or len(set(outputs)) != len(outputs):
            raise ValueError("checkpoint requires unique complete artifacts")
        source = Path(source).resolve()
        artifacts = {}
        for name in outputs:
            path = artifact_path(source, name)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"checkpoint_missing_artifact:{name}")
            artifacts[name] = file_digest(path)
        existing = self.load(unit_id)
        if existing is not None:
            if existing["receipt"]["artifacts"] != artifacts:
                raise ValueError(f"checkpoint_immutable_conflict:{unit_id}")
            return existing
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".pending-{uuid.uuid4().hex}"
        try:
            for name, digest in artifacts.items():
                target = artifact_path(temporary / "artifacts", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with artifact_path(source, name).open("rb") as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                    writer.flush()
                    os.fsync(writer.fileno())
                if file_digest(target) != digest:
                    raise ValueError(f"checkpoint_changed_while_sealing:{unit_id}:{name}")
            _atomic(temporary / "receipt.json", {
                "schema": "research_unit_checkpoint_v1", "unit_id": unit_id,
                "work_binding_id": self.binding, "artifacts": artifacts,
            })
            # One runtime controller owns this namespace. A conflicting unit is
            # an error, not permission to overwrite previously sealed results.
            os.rename(temporary, self._directory(unit_id))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.load(unit_id)
