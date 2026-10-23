"""Make the compiled extension importable from every test module.

Single-config generators (Ninja, Makefiles) put ``_hypergamic`` in ``build/``;
multi-config generators (MSVC) put it in ``build/<config>/``. Both are added so
the tests find it wherever it was built.
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

for _path in (PROJECT_ROOT / "build" / "Release", PROJECT_ROOT / "build", PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
