from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qplatform.point_in_time_universe import (  # noqa: E402
    load_membership_snapshots,
    resolve_asof_membership,
)


def test_asof_membership_never_uses_future_snapshot() -> None:
    snapshots = [
        {"date": "2026-01-02", "symbols": ["000001", "000002"]},
        {"date": "2026-01-06", "symbols": ["000002", "000003"]},
    ]
    membership, audit = resolve_asof_membership(
        snapshots,
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-12",
        ],
        max_age_days=4,
    )
    by_date = {
        date: set(frame["symbol"])
        for date, frame in membership.groupby("date", sort=False)
    }
    assert "2026-01-01" not in by_date
    assert by_date["2026-01-02"] == {"000001", "000002"}
    assert by_date["2026-01-05"] == {"000001", "000002"}
    assert by_date["2026-01-06"] == {"000002", "000003"}
    assert "2026-01-12" not in by_date
    audit_by_date = audit.set_index("date")
    assert audit_by_date.loc["2026-01-01", "reason"] == "before_first_snapshot"
    assert audit_by_date.loc["2026-01-12", "reason"] == "snapshot_too_old"
    assert int(audit_by_date.loc["2026-01-05", "snapshot_age_days"]) == 3


def test_snapshot_loader_normalizes_and_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "dt=2026-01-02" / "cs_norm.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            {"symbol": ["1", "000002", "000002", "bad"]}
        ).to_parquet(path, index=False)
        snapshots = load_membership_snapshots(root)
        assert len(snapshots) == 1
        assert snapshots[0]["date"] == "2026-01-02"
        assert snapshots[0]["symbols"] == ["000001", "000002"]
        assert len(snapshots[0]["sha256"]) == 64


def main() -> int:
    test_asof_membership_never_uses_future_snapshot()
    test_snapshot_loader_normalizes_and_hashes()
    print("point-in-time universe tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
