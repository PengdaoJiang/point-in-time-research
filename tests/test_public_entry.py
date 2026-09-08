from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile

from demo import run


def test_public_demo():
    with tempfile.TemporaryDirectory() as folder:
        result = run(Path(folder) / "case")
        assert result["membership_rejections"] == ["before_first_snapshot", "snapshot_too_old"]
        assert result["feature_audit"]["pass"] is False
        assert result["validation_status"] == "diagnostic_only_causal_signal_path_not_verified"


def test_original_guard_suite():
    path = Path(__file__).with_name("test_research_guard.py")
    spec = importlib.util.spec_from_file_location("original_guard_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
