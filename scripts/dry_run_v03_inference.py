"""
Dry-run V03 inference on one held-out dataset session.

Run from project root after training:
    python scripts/dry_run_v03_inference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_v03 import run_dataset_dry_run  # noqa: E402


def main() -> None:
    result = run_dataset_dry_run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
