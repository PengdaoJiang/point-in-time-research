"""Immutable, indexed result packs. Original artifact bytes and all trials survive.

SQLite is a read-only catalog, not a service. Bounded ZIP parts avoid millions of
filesystem entries; Parquet payloads are stored without another compression pass.
Publication never removes sources or rewrites an existing bundle.
"""
from __future__ import annotations

import hashlib
import contextlib
import json
import os
import re
import sqlite3
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

from .control_io import atomic_json
from .research_readiness import artifact_path, file_digest, read_json


def build_bundle(destination: Path, records, *, max_members=1024, max_part_bytes=64 * 1024**2) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("result_bundle_already_exists")
    if max_members < 1 or max_part_bytes < 1:
        raise ValueError("invalid_result_bundle_part_limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
    pending.mkdir()
    archive, db = None, None
    try:
        db = sqlite3.connect(pending / 'index.sqlite')
        db.execute('CREATE TABLE artifacts (unit TEXT, kind TEXT, name TEXT, part TEXT, member TEXT, '
                   'bytes INTEGER, sha256 TEXT, source TEXT, PRIMARY KEY(unit,kind,name))')
        parts, count, logical_bytes, part_count, part_bytes = {}, 0, 0, 0, 0
        for record in records:
            unit, kind, name = record['unit'], record.get('kind', 'artifact'), record['name']
            if not isinstance(unit, str) or not unit or kind not in {'artifact', 'receipt'}:
                raise ValueError('invalid_result_bundle_identity')
            artifact_path(pending, name)
            source = Path(record['source']).resolve()
            size = source.stat().st_size
            if archive is None or part_count >= max_members or part_bytes + size > max_part_bytes:
                if archive is not None:
                    archive.close()
                    parts[part] = file_digest(pending / part)
                part = f'part-{len(parts):06d}.zip'
                archive = zipfile.ZipFile(pending / part, 'x', allowZip64=True)
                part_count, part_bytes = 0, 0
            member = f'{count:012d}'
            info = zipfile.ZipInfo(member)
            info.compress_type = zipfile.ZIP_STORED if source.suffix.lower() in {'.parquet', '.npz', '.zip', '.gz', '.zst'} else zipfile.ZIP_DEFLATED
            digest, actual_bytes = hashlib.sha256(), 0
            with source.open('rb') as reader, archive.open(info, 'w', force_zip64=True) as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b''):
                    writer.write(block)
                    digest.update(block)
                    actual_bytes += len(block)
            sha = digest.hexdigest()
            if actual_bytes != size or sha != record['sha256']:
                raise ValueError('result_bundle_source_changed')
            db.execute('INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?)',
                       (unit, kind, name, part, member, size, sha, str(source)))
            count += 1
            logical_bytes += size
            part_count += 1
            part_bytes += size
        if archive is None:
            raise ValueError('empty_result_bundle')
        archive.close()
        archive = None
        parts[part] = file_digest(pending / part)
        units = db.execute('SELECT COUNT(DISTINCT unit) FROM artifacts').fetchone()[0]
        db.commit()
        db.close()
        db = None
        manifest = {'schema': 'research_result_bundle_v1', 'artifacts': count, 'units': units,
                    'logical_bytes': logical_bytes, 'parts': parts, 'index_sha256': file_digest(pending / 'index.sqlite'),
                    'source_retirement': 'not_authorized_by_bundle_creation'}
        atomic_json(pending / 'manifest.json', manifest)
        for path in pending.iterdir():
            with path.open('rb+') as stream:
                os.fsync(stream.fileno())
        with ResultBundle(pending / 'manifest.json') as bundle:
            bundle.verify_all()
        os.rename(pending, destination)
        return {'manifest': str(destination / 'manifest.json'), 'sha256': file_digest(destination / 'manifest.json'),
                'files': len(parts) + 2, 'stored_bytes': sum(p.stat().st_size for p in destination.iterdir()), **manifest}
    finally:
        if archive is not None:
            archive.close()
        if db is not None:
            db.close()
        # Failed staging is retained for explicit recovery, never accepted as a bundle.


class ResultBundle:
    def __init__(self, manifest: Path, expected_sha256: str | None = None, *, max_open_parts: int = 8):
        if type(max_open_parts) is not int or max_open_parts < 1:
            raise ValueError('invalid_result_bundle_handle_limit')
        self.path = Path(manifest).resolve()
        if expected_sha256 and file_digest(self.path) != expected_sha256:
            raise ValueError('result_bundle_manifest_mismatch')
        self.root = self.path.parent
        self.manifest = read_json(self.path)
        if self.manifest.get('schema') != 'research_result_bundle_v1':
            raise ValueError('unknown_result_bundle_schema')
        index = self.root / 'index.sqlite'
        if file_digest(index) != self.manifest['index_sha256']:
            raise ValueError('result_bundle_index_mismatch')
        self.db = sqlite3.connect(index.as_uri() + '?mode=ro&immutable=1', uri=True)
        self.verified = {}
        self.verified_members = {}
        self.max_open_parts = max_open_parts
        self._archives = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()
        self.db.close()

    def _part(self, name):
        if not re.fullmatch(r'part-[0-9]{6}\.zip', name) or name not in self.manifest['parts']:
            raise ValueError('invalid_result_bundle_part')
        path = artifact_path(self.root, name)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.verified.get(name) != signature:
            archive = self._archives.pop(name, None)
            if archive is not None:
                archive.close()
            if file_digest(path) != self.manifest['parts'][name]:
                raise ValueError('result_bundle_part_mismatch')
            self.verified[name] = signature
        return path

    def _archive(self, name):
        path = self._part(name)
        if name not in self._archives:
            while len(self._archives) >= self.max_open_parts:
                _, oldest = self._archives.popitem(last=False)
                oldest.close()
            self._archives[name] = zipfile.ZipFile(path)
        self._archives.move_to_end(name)
        return self._archives[name]

    def entries(self):
        return self.db.execute('SELECT unit,kind,name,part,member,bytes,sha256 FROM artifacts ORDER BY part,member')

    def unit_receipt(self, unit: str, binding: str | None = None):
        payload = self.read_bytes(unit, 'receipt.json', kind='receipt')
        receipt = json.loads(payload)
        if (receipt.get('schema') != 'research_unit_checkpoint_v1'
                or receipt.get('unit_id') != unit
                or (binding is not None and receipt.get('work_binding_id') != binding)
                or not receipt.get('artifacts')):
            raise ValueError('bundle_checkpoint_identity_mismatch')
        actual = dict(self.db.execute(
            "SELECT name,sha256 FROM artifacts WHERE unit=? AND kind='artifact'", (unit,)))
        receipts = list(self.db.execute(
            "SELECT name FROM artifacts WHERE unit=? AND kind='receipt'", (unit,)))
        if actual != receipt['artifacts'] or receipts != [('receipt.json',)]:
            raise ValueError('bundle_checkpoint_coverage_mismatch')
        for name in actual:
            artifact_path(self.root, name)
        return {'receipt': receipt, 'receipt_sha256': hashlib.sha256(payload).hexdigest()}

    @contextlib.contextmanager
    def open_artifact(self, unit: str, name: str, *, kind='artifact'):
        row = self.db.execute('SELECT part,member,bytes,sha256 FROM artifacts WHERE unit=? AND kind=? AND name=?',
                              (unit, kind, name)).fetchone()
        if row is None:
            raise KeyError((unit, kind, name))
        archive = self._archive(row[0])
        signature = self.verified[row[0]]
        if self.verified_members.get((row[0], row[1])) != signature:
            digest, size = hashlib.sha256(), 0
            with archive.open(row[1]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
            if size != row[2] or digest.hexdigest() != row[3]:
                raise ValueError('result_bundle_artifact_mismatch')
            self.verified_members[(row[0], row[1])] = signature
        with archive.open(row[1]) as stream:
            yield stream

    def read_bytes(self, unit: str, name: str, *, kind='artifact') -> bytes:
        row = self.db.execute('SELECT part,member,bytes,sha256 FROM artifacts WHERE unit=? AND kind=? AND name=?',
                              (unit, kind, name)).fetchone()
        if row is None:
            raise KeyError((unit, kind, name))
        data = self._archive(row[0]).read(row[1])
        if len(data) != row[2] or hashlib.sha256(data).hexdigest() != row[3]:
            raise ValueError('result_bundle_artifact_mismatch')
        return data

    def read_table(self, unit: str, name: str, *, columns=None):
        import pyarrow.parquet as pq
        # Integrity verification streams bytes once; projection never buffers the
        # entire member in Python memory. First verification still reads all bytes.
        with self.open_artifact(unit, name) as stream:
            return pq.ParquetFile(stream).read(columns=columns, use_threads=False)

    def restore_unit(self, unit: str, destination: Path, *, binding: str | None = None):
        destination = Path(destination).resolve()
        if destination.exists():
            raise ValueError('result_restore_destination_exists')
        loaded = self.unit_receipt(unit, binding)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
        pending.mkdir()
        # A failed recovery leaves explicit staging, never a partial published unit.
        for name, sha in loaded['receipt']['artifacts'].items():
            target = artifact_path(pending / 'artifacts', name)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with self.open_artifact(unit, name) as reader, target.open('xb') as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b''):
                    digest.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != sha:
                raise ValueError('result_restore_hash_mismatch')
        with (pending / 'receipt.json').open('xb') as writer:
            writer.write(self.read_bytes(unit, 'receipt.json', kind='receipt'))
            writer.flush()
            os.fsync(writer.fileno())
        os.rename(pending, destination)
        return {'status': 'restored', 'unit': unit, 'destination': str(destination),
                'receipt_sha256': loaded['receipt_sha256'], 'artifacts': len(loaded['receipt']['artifacts'])}

    def verify_all(self):
        count, logical_bytes, units = 0, 0, set()
        current_part, archive = None, None
        for unit, kind, name, part, member, size, sha in self.entries():
            if part != current_part:
                archive = self._archive(part)
                current_part = part
            digest, total = hashlib.sha256(), 0
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    total += len(block)
            if total != size or digest.hexdigest() != sha:
                raise ValueError('result_bundle_artifact_mismatch')
            self.verified_members[(part, member)] = self.verified[part]
            count += 1
            logical_bytes += total
            units.add(unit)
        if (count, len(units), logical_bytes) != tuple(self.manifest[k] for k in ('artifacts', 'units', 'logical_bytes')):
            raise ValueError('result_bundle_coverage_mismatch')
        return {'status': 'pass', 'artifacts': count, 'units': len(units), 'logical_bytes': logical_bytes}
