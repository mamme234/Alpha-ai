#!/usr/bin/env python3
"""Run the AlphaAI evaluation suite and write evaluation/reports/<timestamp>.json.

Evaluators report measured numbers or an explicit SKIPPED reason — never a
placeholder score.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("evaluate", description=__doc__))
