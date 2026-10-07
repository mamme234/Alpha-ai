#!/usr/bin/env python3
"""Report the state of the AlphaAI training foundation: datasets, tokenizer,
checkpoints, tracked runs and the latest evaluation reports."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("status", description=__doc__))
