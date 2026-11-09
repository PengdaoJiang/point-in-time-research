import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from qplatform.research_readiness import file_digest
from qplatform.research_result_bundle import ResultBundle, build_bundle


class StorageOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


    def records(self):
        records = []
        for n in range(5):
            source = self.root / f'input{n}.json'
            source.write_text(json.dumps({'return': n-3, 'negative_is_retained': True, 'padding': 'x'*1000}))
            records.append({'unit': str(n), 'name': 'result.json', 'source': source, 'sha256': file_digest(source)})
        table = self.root / 'ledger.parquet'
        pd.DataFrame({'symbol': ['000001', '688001'], 'pnl': [-3.0, 7.0]}).to_parquet(table)
        records.append({'unit': '0', 'name': 'ledger.parquet', 'source': table, 'sha256': file_digest(table)})
        return records

    def test_bundle_exact_bytes_types_negative_cases_and_rotation(self):
        records = self.records()
        result = build_bundle(self.root / 'bundle', records, max_members=2)
        self.assertEqual(len(result['parts']), 3)
        with ResultBundle(Path(result['manifest']), result['sha256']) as bundle:
            self.assertEqual(bundle.verify_all()['artifacts'], len(records))
            for record in records:
                self.assertEqual(bundle.read_bytes(record['unit'], record['name']), record['source'].read_bytes())
            pd.testing.assert_frame_equal(bundle.read_table('0', 'ledger.parquet').to_pandas(),
                                          pd.read_parquet(self.root / 'ledger.parquet'))
            self.assertEqual(bundle.read_table('0', 'ledger.parquet', columns=['pnl']).column_names, ['pnl'])
            with self.assertRaises(KeyError):
                bundle.read_bytes('missing', 'result.json')
        self.assertTrue(all(r['source'].is_file() for r in records))
        with self.assertRaisesRegex(ValueError, 'already_exists'):
            build_bundle(self.root / 'bundle', records)

    def test_changed_source_never_publishes(self):
        records = self.records()
        records[-1]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source_changed'):
            build_bundle(self.root / 'broken', records)
        self.assertFalse((self.root / 'broken').exists())
        self.assertTrue(all(r['source'].is_file() for r in records))

    def test_corrupt_part_and_index_fail_closed(self):
        result = build_bundle(self.root / 'bundle', self.records())
        with ResultBundle(Path(result['manifest'])) as bundle:
            bundle.read_bytes('0', 'result.json')
            part = self.root / 'bundle' / next(iter(result['parts']))
            with part.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'part_mismatch'):
                bundle.read_bytes('0', 'result.json')
        index = self.root / 'bundle/index.sqlite'
        index.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'index_mismatch'):
            ResultBundle(Path(result['manifest']))

    def test_duplicate_identity_never_publishes(self):
        records = self.records()
        with self.assertRaises(sqlite3.IntegrityError):
            build_bundle(self.root / 'bad', records + [records[0]])
        self.assertFalse((self.root / 'bad').exists())



if __name__ == '__main__':
    unittest.main()
