import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isolated database for API tests (set before the app modules are imported)
_tmp = tempfile.mkdtemp(prefix="smartyield_test_")
os.environ.setdefault("SMARTYIELD_DB", str(Path(_tmp) / "test.db"))
