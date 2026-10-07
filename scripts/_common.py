"""Shared bootstrap for the AlphaAI training and evaluation scripts.

The scripts are thin wrappers: they parse arguments and call the same
``alphaai.training`` functions the CLI uses, so both entry points always behave
identically (and are covered by the same tests).

Run them with the repository on ``sys.path`` (they add it themselves):

    python scripts/training/train_tokenizer.py --dataset alphaai-sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_common(description: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", help="Dataset name under datasets/ (default: config).")
    parser.add_argument("--tokenizer", help="Tokenizer directory or name (default: tokenizer/).")
    parser.add_argument("--experiment", help="Experiment name (default: config).")
    parser.add_argument("--project-root", help="AlphaAI project root (default: cwd).")
    parser.add_argument("--json", action="store_true", help="Emit JSON (scripts always do).")
    return parser.parse_args(argv)


def run(action: str, argv: Sequence[str] | None = None, *, description: str = "") -> int:
    """Load the AlphaAI config, run a training/evaluation action, print the report."""

    args = parse_common(description or f"AlphaAI {action}", argv)
    from alphaai.training import ACTIONS, _config

    config = _config(args)
    payload = ACTIONS[action](config, args)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if payload.get("ok", False) else 1
