from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Test doubles are only honoured in test mode (see jarvis.platform.factory).
os.environ["JARVIS_TEST"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "jarvis-data"
    monkeypatch.setenv("JARVIS_DATA_DIR", str(d))
    return d


@pytest.fixture
def cfg(data_dir):
    from jarvis.common.config import load_config

    return load_config(environ={})
