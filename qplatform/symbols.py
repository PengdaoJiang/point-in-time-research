from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


_DIGIT_RE = re.compile(r"\d+")
_SYMBOL6_RE = re.compile(r"^\d{6}$")


def normalize_symbol(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return ""

    if isinstance(value, int):
        return str(value).zfill(6)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        iv = int(round(value))
        if abs(value - iv) < 1e-9:
            return str(iv).zfill(6)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if len(s) == 6 and s.isdigit():
            return s
        low = s.lower()
        if low in {"nan", "none", "null"}:
            return ""
        if s.isdigit():
            return s[-6:].zfill(6)
        if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
            try:
                return str(int(float(s))).zfill(6)
            except ValueError:
                return ""
    else:
        s = str(value).strip()
    if not s:
        return ""
    low = s.lower()
    if low in {"nan", "none", "null"}:
        return ""

    if s.lstrip("+-").isdigit():
        try:
            return str(int(s)).zfill(6)
        except ValueError:
            return ""

    if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
        try:
            return str(int(float(s))).zfill(6)
        except ValueError:
            return ""

    digit_chunks = _DIGIT_RE.findall(s)
    if not digit_chunks:
        return ""
    digits = "".join(digit_chunks)
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6)


def is_symbol6(value: Any) -> bool:
    return bool(_SYMBOL6_RE.fullmatch(str(value).strip()))


def assert_symbol6_series(series: pd.Series, *, context: str = "symbol") -> None:
    if series is None:
        raise ValueError(f"{context}: symbol series is None")
    s = series.astype(str).str.strip()
    bad = ~s.str.fullmatch(r"\d{6}")
    if bool(bad.any()):
        sample = s.loc[bad].head(10).tolist()
        raise ValueError(f"{context}: found non-6-digit symbol values before write, samples={sample}")
