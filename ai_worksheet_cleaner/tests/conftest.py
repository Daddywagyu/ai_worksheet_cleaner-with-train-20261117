import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from tests import worksheet as ws


@pytest.fixture(scope="session")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws")
    pdf, clean, dirty, gt = ws.make_pair(d)
    return pdf, clean, dirty, gt
