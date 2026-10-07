#!/usr/bin/env python3
"""Validate every AlphaAI dataset (records, schema, split hashes, manifest)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("validate", description=__doc__))
