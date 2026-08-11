"""Seeding, logging, checkpoint IO."""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def results_dir(run_name: str, root: str | Path = "results") -> Path:
    d = Path(root) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class TSVLogger:
    path: Path
    columns: Iterable[str] = field(default_factory=list)
    _wrote_header: bool = field(default=False, init=False)

    def log(self, row: Dict[str, Any]) -> None:
        cols = list(self.columns) if self.columns else list(row.keys())
        mode = "a" if self._wrote_header or self.path.exists() else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            if not self._wrote_header and mode == "w":
                writer.writeheader()
                self._wrote_header = True
            writer.writerow({c: row.get(c, "") for c in cols})
        self._wrote_header = True
