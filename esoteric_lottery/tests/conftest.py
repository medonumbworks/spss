import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import event_of, subject_of  # noqa: E402
from methods.common import games_from_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def subject(cfg):
    return subject_of(cfg)


@pytest.fixture(scope="session")
def event(cfg):
    return event_of(cfg)


@pytest.fixture(scope="session")
def games(cfg):
    return games_from_config(cfg)


@pytest.fixture(scope="session")
def root():
    return ROOT
