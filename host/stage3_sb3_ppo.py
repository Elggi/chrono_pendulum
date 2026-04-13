#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from rl_train.train_ppo import main

if __name__ == "__main__":
    main()
