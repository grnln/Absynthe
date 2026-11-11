"""pytest configuration: make the repository root importable.

The tests import the framework packages (``graph``, ``motifs``, ``utils``,
``interfaces``) directly, so the repository root must be on ``sys.path``
regardless of the directory pytest is launched from.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
