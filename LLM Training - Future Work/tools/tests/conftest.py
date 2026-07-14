"""Every module under data_collection/, evaluation/, and deployment/ uses
flat sibling imports (e.g. `from inter_rater_reliability import ...`)
rather than package-relative imports, so each directory needs to be on
sys.path for both direct script execution and for these tests to import
them.
"""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent
for _subdir in ("data_collection", "evaluation", "deployment"):
    _path = str(_TOOLS_DIR / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)
