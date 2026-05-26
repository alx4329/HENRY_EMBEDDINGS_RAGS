"""Bootstrap común a todos los scripts.

Agrega ``src`` al ``sys.path`` cuando los scripts se ejecutan sin haber
hecho ``pip install -e .`` (modo desarrollo rápido con ``uv run``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
