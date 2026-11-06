#!/usr/bin/env python3
"""Thin CLI for the canonical frozen WebShop goal converter."""
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentflow"))

from agentflow.trace.webshop_goal import main


if __name__ == "__main__":
    raise SystemExit(main())
