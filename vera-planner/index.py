"""Vercel entrypoint for vera-planner (auto-detected by the Vercel FastAPI preset)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("VERA_NAMESPACE", "vera-planner")

from strategy import PlannerStrategy  # noqa: E402
from vera.api import create_app  # noqa: E402

app = create_app(PlannerStrategy(), version="1.0.0")
