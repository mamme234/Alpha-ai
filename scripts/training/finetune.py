#!/usr/bin/env python3
"""Run the AlphaAI reference training loop (real optimiser steps, real checkpoints).

This is the *reference* pipeline: it trains a small decoder-only Transformer on a
CPU so the whole training foundation can be exercised end to end. It is honest
about scale — see configs/training/alpha-1-pretrain.toml and docs/TRAINING.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("finetune", description=__doc__))
