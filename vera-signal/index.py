"""Vercel entrypoint for vera-signal (auto-detected by the Vercel FastAPI preset)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("VERA_NAMESPACE", "vera-signal")

from strategy import SignalStrategy  # noqa: E402
from vera.api import create_app  # noqa: E402

app = create_app(SignalStrategy(), version="1.0.0")
