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


class CheckpointStore:
    def __init__(self, root: Path, work_binding_id: str):
        if not re.fullmatch(r"[0-9a-f]{64}", work_binding_id):
            raise ValueError("checkpoint requires a runtime work binding")
        self.binding = work_binding_id
        self.root = Path(root).resolve() / work_binding_id

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
            return None
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
