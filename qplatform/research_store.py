"""Immutable research tables, shared by content rather than copied per run.

This is an additive research read plane, not a replacement for source ingestion.
Snapshots preserve rows and types; date partitioning changes physical row order.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .control_io import atomic_json
from .research_readiness import file_digest
from .research_runtime import _lock


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _date(value):
    from datetime import date
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("date must be YYYY-MM-DD")
    return value


def freeze_table(store: Path, sources: list[Path], semantics: dict, *,
                 partition_by_date: bool = False, max_materialize_bytes: int = 512 * 1024**2,
                 expected_source_hashes: list[str] | None = None) -> dict:
    """Publish only after stable source hashes and exact schemas have been checked."""
    store = Path(store).resolve()
    if any(part.lower() in {"cross_section", "minute_raw", "day_raw", ".trash", "archive", "review_packages"} for part in store.parts):
        raise ValueError("protected source/archive directory cannot be a research store")
    sources = [Path(p).resolve() for p in sources]
    required = {"universe", "price_basis", "frequency", "availability", "role"}
    if not required.issubset(semantics) or not all(isinstance(semantics[k], str) and semantics[k].strip() for k in required):
        raise ValueError("declare universe, price_basis, frequency, availability and role")
    if not sources or len(set(sources)) != len(sources) or max_materialize_bytes <= 0:
        raise ValueError("unique explicit sources and positive materialization budget required")
    if expected_source_hashes is not None and (
            len(expected_source_hashes) != len(sources)
            or any(not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)
                   for sha in expected_source_hashes)):
        raise ValueError('expected_source_hashes_must_match_source_order')
    if any(any(part.lower() in {".trash", "archive", "review_packages"} for part in p.parts) for p in sources):
        raise ValueError("historical/archive sources require a separate explicit recovery workflow")
    for path in sources:
        if path.is_relative_to(store) or store.is_relative_to(path) or path.suffix != ".parquet":
            raise ValueError("sources must be external Parquet files")
    # Shared-store publication is serialized. Readers use only immutable manifests.
    with _lock(store), tempfile.TemporaryDirectory(prefix=".freeze-", dir=store) as temp:
        temp = Path(temp)
        entries, provenance, signatures, schema = [], [], [], None
        for index, source in enumerate(sources):
            before = _signature(source)
            provenance.append({"path": str(source), "sha256": file_digest(source), "bytes": before[0]})
            if _signature(source) != before:
                raise ValueError("source changed during fingerprint")
            if expected_source_hashes is not None and provenance[-1]['sha256'] != expected_source_hashes[index]:
                raise ValueError('source_differs_from_expected_binding')
            signatures.append(before)
        layout = "date_partitions" if partition_by_date else "source_files"
        request = {"schema": "research_table_build_v1", "sources": provenance, "semantics": semantics,
                   "layout": layout, "arrow_version": pa.__version__}
        build_id = hashlib.sha256(_encoded(request)).hexdigest()
        build_path = store / "builds" / f"{build_id}.json"
        if build_path.exists():
            build = json.loads(build_path.read_bytes())
            snapshot_id = build["snapshot_id"]
            if build.get("request") != request or not re.fullmatch("[0-9a-f]{64}", snapshot_id):
                raise ValueError("invalid snapshot build index")
            target = store / "snapshots" / f"{snapshot_id}.json"
            cached = ResearchTable(target)
            if any(cached.manifest[k] != request[k] for k in ("sources", "semantics", "layout")):
                raise ValueError("snapshot build index mismatch")
            cached.verify_all()
            for source, ref, signature in zip(sources, provenance, signatures):
                if _signature(source) != signature or file_digest(source) != ref["sha256"]:
                    raise ValueError("source set changed during freeze")
            return {"snapshot": str(target), "snapshot_id": snapshot_id, "rows": cached.manifest["rows"],
                    "objects": len(cached.manifest["objects"]), "added_bytes": 0,
                    "reused_bytes": sum(e["bytes"] for e in cached.manifest["objects"]), "freeze_cache_hit": True}
        added_bytes, reused_bytes = 0, 0

        def publish(path, source_index, date_value=None):
            nonlocal added_bytes, reused_bytes
            sha = file_digest(path)
            target = store / "objects" / sha[:2] / f"{sha}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            size = path.stat().st_size
            if target.exists():
                if file_digest(target) != sha:
                    raise ValueError("corrupt shared object; refusing overwrite")
                reused_bytes += size
            else:
                # temp and destination are on the same volume; never hardlink live inputs.
                with path.open("rb+") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
                os.rename(path, target)
                added_bytes += size
            with pq.ParquetFile(target) as metadata_file:
                metadata = metadata_file.metadata
            entries.append({"sha256": sha, "bytes": size, "rows": metadata.num_rows,
                            "source_index": source_index, "date": date_value})

        for index, source in enumerate(sources):
            before, source_sha = signatures[index], provenance[index]["sha256"]
            with pq.ParquetFile(source) as parquet:
                if schema is None:
                    schema = parquet.schema_arrow
                elif not schema.equals(parquet.schema_arrow, check_metadata=True):
                    raise ValueError("source schemas differ; explicit normalization required")
                if partition_by_date:
                    if "date" not in schema.names or not (pa.types.is_string(schema.field("date").type) or pa.types.is_large_string(schema.field("date").type)):
                        raise ValueError("date partitioning requires a string date column")
                    decoded = sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups))
                    if decoded > max_materialize_bytes:
                        raise ValueError("source exceeds materialization budget; shard upstream or raise explicit budget")
                    table = parquet.read(use_threads=False)
                    values = pc.unique(table["date"]).to_pylist()
                    if None in values:
                        raise ValueError("null date cannot be partitioned")
                    for value in sorted(values):
                        _date(value)
                        part = table.filter(pc.equal(table["date"], value))
                        path = temp / "part.parquet"
                        pq.write_table(part, path, compression="zstd", row_group_size=65536)
                        publish(path, index, value)
                    if not values:  # Keep the typed empty dataset, not an empty file list.
                        path = temp / "empty.parquet"
                        pq.write_table(table, path)
                        publish(path, index)
                else:
                    path = temp / "copy.parquet"
                    shutil.copyfile(source, path)
                    if file_digest(path) != source_sha:
                        raise ValueError("source changed during copy")
                    publish(path, index)
            if _signature(source) != before or file_digest(source) != source_sha:
                raise ValueError("source changed during freeze; no snapshot published")
        # Detect changes to earlier members of a multi-file input set as well.
        for source, ref in zip(sources, provenance):
            if file_digest(source) != ref["sha256"]:
                raise ValueError("source set changed during freeze")
        manifest = {
            "schema": "research_table_snapshot_v1", "semantics": semantics,
            "layout": layout,
            "arrow_schema": base64.b64encode(schema.serialize().to_pybytes()).decode(),
            "sources": provenance, "objects": entries,
            "rows": sum(e["rows"] for e in entries),
        }
        payload = _encoded(manifest)
        snapshot_id = hashlib.sha256(payload).hexdigest()
        target = store / "snapshots" / f"{snapshot_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("snapshot immutable conflict")
        else:
            pending = temp / "manifest.json"
            with pending.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(pending, target)
        atomic_json(build_path, {"request": request, "snapshot_id": snapshot_id})
        return {"snapshot": str(target), "snapshot_id": snapshot_id, "rows": manifest["rows"],
                "objects": len(entries), "added_bytes": added_bytes, "reused_bytes": reused_bytes,
                "freeze_cache_hit": False}


class ResearchTable:
    """One verified handle, reusable for many projected/filtered scans.

    Selected immutable objects are hash-verified once per handle, with stat
    changes invalidating that verification. verify_all() is the runtime gate.
    """
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.verified = {}
        self.manifest = None
        if self.path.suffix == ".json":
            payload = self.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.path.stem:
                raise ValueError("snapshot manifest hash mismatch")
            self.manifest = json.loads(payload)
            if self.manifest.get("schema") != "research_table_snapshot_v1":
                raise ValueError("unsupported research snapshot")
            self.schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(self.manifest["arrow_schema"])))
            self.store = self.path.parent.parent
            if self.path.parent.name != "snapshots":
                raise ValueError("snapshot must reside in its store")
        else:
            self.schema = pq.ParquetFile(self.path).schema_arrow

    def _object(self, entry):
        sha = entry["sha256"]
        if not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError("invalid object hash")
        path = self.store / "objects" / sha[:2] / f"{sha}.parquet"
        if not path.resolve().is_relative_to(self.store):
            raise ValueError("object escapes store")
        signature = _signature(path)
        if self.verified.get(sha) != signature:
            if signature[0] != entry["bytes"] or file_digest(path) != sha:
                raise ValueError(f"snapshot object mismatch:{sha}")
            self.verified[sha] = signature
        return path

    def verify_all(self):
        if self.manifest:
            for entry in self.manifest["objects"]:
                self._object(entry)
        return {"status": "verified", "objects": len(self.verified)}

    def scanner(self, *, columns=None, dates=None, symbols=None, batch_size=65536):
        columns = list(columns) if columns is not None else None
        if columns is not None and (not columns or len(set(columns)) != len(columns) or set(columns) - set(self.schema.names)):
            raise ValueError("projection contains missing/duplicate columns or is empty")
        predicates = []
        if dates is not None:
            dates = [_date(d) for d in dates]
            predicates.append(ds.field("date").isin(pa.array(dates, type=self.schema.field("date").type)))
        if symbols is not None:
            if any(not isinstance(s, str) or not re.fullmatch("[0-9]{6}", s) for s in symbols):
                raise ValueError("symbols must be exact six-digit strings")
            predicates.append(ds.field("symbol").isin(pa.array(symbols, type=self.schema.field("symbol").type)))
        predicate = None
        for item in predicates:
            predicate = item if predicate is None else predicate & item
        entries = self.manifest["objects"] if self.manifest else []
        selected = [e for e in entries if dates is None or e["date"] is None or e["date"] in dates]
        paths = [str(self._object(e)) for e in selected] if self.manifest else [str(self.path)]
        dataset = ds.dataset(paths, schema=self.schema, format="parquet") if paths else ds.dataset(pa.Table.from_batches([], schema=self.schema))
        self.last_scan = {"files_selected": len(paths), "files_total": len(entries) if self.manifest else 1,
                          "object_bytes_selected": sum(e["bytes"] for e in selected) if self.manifest else self.path.stat().st_size,
                          "columns_selected": len(columns or self.schema.names), "columns_total": len(self.schema),
                          "note": "object bytes are not measured physical IO; row and column pushdown also apply"}
        return dataset.scanner(columns=columns, filter=predicate, batch_size=batch_size, use_threads=False)

    def read(self, **query):
        return self.scanner(**query).to_table().to_pandas()


def read_research_table(path: Path, **query):
    return ResearchTable(path).read(**query)
