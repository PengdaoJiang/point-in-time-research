from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .symbols import normalize_symbol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_membership_snapshots(snapshot_root: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for path in sorted(Path(snapshot_root).glob("dt=*/cs_norm.parquet")):
        snapshot_date = path.parent.name.removeprefix("dt=")
        pd.Timestamp(snapshot_date)
        if snapshot_date in seen_dates:
            raise ValueError(f"duplicate membership snapshot date: {snapshot_date}")
        seen_dates.add(snapshot_date)
        frame = pd.read_parquet(path, columns=["symbol"])
        symbols = sorted(
            {
                symbol
                for value in frame["symbol"]
                if (symbol := normalize_symbol(value))
            }
        )
        snapshots.append(
            {
                "date": snapshot_date,
                "path": path,
                "sha256": sha256_file(path),
                "symbols": symbols,
            }
        )
    if not snapshots:
        raise FileNotFoundError(
            f"no normalized membership snapshots under {snapshot_root}"
        )
    return snapshots


def resolve_asof_membership(
    snapshots: list[dict[str, Any]],
    decision_dates: Iterable[str],
    *,
    max_age_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(max_age_days) < 0:
        raise ValueError("max_age_days must be non-negative")
    ordered = sorted(snapshots, key=lambda item: str(item["date"]))
    snapshot_dates = [pd.Timestamp(item["date"]) for item in ordered]
    if len(snapshot_dates) != len(set(snapshot_dates)):
        raise ValueError("membership snapshot dates must be unique")

    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    selected_index = -1
    for decision_date in sorted({str(value) for value in decision_dates}):
        decision_ts = pd.Timestamp(decision_date)
        while (
            selected_index + 1 < len(snapshot_dates)
            and snapshot_dates[selected_index + 1] <= decision_ts
        ):
            selected_index += 1
        if selected_index < 0:
            audits.append(
                {
                    "date": decision_date,
                    "covered": False,
                    "snapshot_date": "",
                    "snapshot_age_days": np.nan,
                    "membership_symbol_count": 0,
                    "reason": "before_first_snapshot",
                }
            )
            continue

        selected = ordered[selected_index]
        age = int((decision_ts - snapshot_dates[selected_index]).days)
        if age > int(max_age_days):
            audits.append(
                {
                    "date": decision_date,
                    "covered": False,
                    "snapshot_date": str(selected["date"]),
                    "snapshot_age_days": age,
                    "membership_symbol_count": int(len(selected["symbols"])),
                    "reason": "snapshot_too_old",
                }
            )
            continue

        rows.append(
            pd.DataFrame(
                {
                    "date": decision_date,
                    "symbol": selected["symbols"],
                    "membership_snapshot_date": str(selected["date"]),
                    "membership_snapshot_age_days": age,
                }
            )
        )
        audits.append(
            {
                "date": decision_date,
                "covered": True,
                "snapshot_date": str(selected["date"]),
                "snapshot_age_days": age,
                "membership_symbol_count": int(len(selected["symbols"])),
                "reason": "",
            }
        )

    membership = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "membership_snapshot_date",
                "membership_snapshot_age_days",
            ]
        )
    )
    audit = pd.DataFrame(audits)
    if not membership.empty:
        future = pd.to_datetime(
            membership["membership_snapshot_date"]
        ) > pd.to_datetime(membership["date"])
        if bool(future.any()):
            raise RuntimeError("future membership snapshot selected")
    return membership, audit
