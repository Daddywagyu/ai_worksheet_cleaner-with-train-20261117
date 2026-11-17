"""Test 7: no network code anywhere in the application."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(requests|urllib|http\.client|httpx|socket|ftplib|smtplib|aiohttp|websockets)\b", re.M)


def test_no_network_imports():
    offenders = []
    for p in ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        if FORBIDDEN.search(p.read_text(encoding="utf-8")):
            offenders.append(str(p))
    assert not offenders, offenders


def test_no_urls_called():
    for p in ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        assert "http://" not in p.read_text(encoding="utf-8").replace("# ", "") or p.name == "main_window.py" is False
