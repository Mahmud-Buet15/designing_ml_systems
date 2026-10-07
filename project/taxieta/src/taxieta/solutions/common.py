"""Helpers shared by the solution modules: consistent report files and small utilities."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from taxieta import config


def out_dir() -> Path:
    d = config.REPORTS_DIR / "solutions"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Report:
    """Collects markdown sections + a JSON payload; `save()` writes reports/solutions/<name>.*"""

    def __init__(self, name: str, title: str):
        self.name = name
        self.parts: list[str] = [f"# {title}\n"]
        self.data: dict = {}

    def h(self, text: str) -> None:
        self.parts.append(f"\n## {text}\n")

    def p(self, text: str) -> None:
        self.parts.append(f"{text}\n")
        print(text)

    def table(self, df: pd.DataFrame, key: str | None = None, floatfmt: str = ".3f") -> None:
        md = df.to_markdown(index=False, floatfmt=floatfmt) if len(df) else "_(empty)_"
        self.parts.append(md + "\n")
        print(df.to_string(index=False))
        if key:
            self.data[key] = json.loads(df.to_json(orient="records", date_format="iso"))

    def put(self, key: str, value) -> None:
        self.data[key] = value

    def save(self) -> Path:
        d = out_dir()
        (d / f"{self.name}.md").write_text("\n".join(self.parts))
        (d / f"{self.name}.json").write_text(json.dumps(self.data, indent=2, default=_default))
        print(f"\nwrote {d / (self.name + '.md')}")
        return d / f"{self.name}.md"


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


def month_key(df: pd.DataFrame) -> pd.Series:
    return df["pickup_ts"].dt.year * 100 + df["pickup_ts"].dt.month


def in_months(df: pd.DataFrame, months: list[tuple[int, int]]) -> pd.DataFrame:
    return df[month_key(df).isin([y * 100 + m for y, m in months])]


def sample_frac_used(year: int, month: int) -> float:
    """The --sample-frac used at ingest (to scale counts back to the full population)."""
    p = config.REPORTS_DIR / "ingest" / f"{year}-{month:02d}.json"
    if p.exists():
        frac = json.loads(p.read_text()).get("sample_frac")
        return float(frac) if frac else 1.0
    return 1.0
