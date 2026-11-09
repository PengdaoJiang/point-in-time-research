"""Offline synthetic example: immutable data, reuse, projection and mutation rejection."""
import json
from pathlib import Path
import tempfile

import pandas as pd

from qplatform.research_store import ResearchTable, freeze_table


def main():
    with tempfile.TemporaryDirectory(prefix='research-data-example-') as temporary:
        root = Path(temporary)
        source = root / 'input.parquet'
        frame = pd.DataFrame({'symbol': ['TEST_A', 'TEST_B', 'TEST_A'],
                              'date': ['2026-01-01', '2026-01-01', '2026-01-02'],
                              'value': [1.0, 2.0, 3.0], 'unused': [0, 0, 0]})
        frame.to_parquet(source, index=False)
        semantics = {'universe': 'synthetic', 'price_basis': 'not_prices',
                     'frequency': 'daily', 'availability': 'fixture dates only', 'role': 'example'}
        first = freeze_table(root / 'store', [source], semantics, partition_by_date=True)
        again = freeze_table(root / 'store', [source], semantics, partition_by_date=True)
        reader = ResearchTable(Path(first['snapshot']))
        selected = reader.read(columns=['symbol', 'value'], dates=['2026-01-02'])
        assert selected['value'].tolist() == [3.0]
        assert again['added_bytes'] == 0 and again['snapshot_id'] == first['snapshot_id']
        frame.assign(value=99.0).to_parquet(source, index=False)
        assert reader.read()['value'].tolist() == [1.0, 2.0, 3.0]
        changed = freeze_table(root / 'store', [source], semantics, partition_by_date=True)
        assert changed['snapshot_id'] != first['snapshot_id']
        print(json.dumps({'synthetic_only': True, 'reused_snapshot': True,
                          'source_mutation_isolated': True, 'projected_rows': len(selected),
                          'original_snapshot': first['snapshot_id'],
                          'changed_snapshot': changed['snapshot_id']}, indent=2))


if __name__ == '__main__': main()
