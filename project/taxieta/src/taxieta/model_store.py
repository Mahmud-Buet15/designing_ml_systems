"""A minimal, file-based model store (Milestones 5, 8, 9; Ch. 10 "Model Store").

data/models/
    v0003/
        model.joblib        # model definition + parameters
        hist.joblib         # fitted HistoricalStats (part of featurization state)
        heuristic.joblib    # fallback model shipped with every version
        meta.json           # data window, features, params, metrics, git commit, deps, tags, parent
    champion.json           # {"version": "v0003"}

Ch. 10 lists eight artifacts a model store should keep. meta.json covers most of them.
TODO(M9): check which of the eight are still missing (hint: container image tag,
experiment artifacts) and add them -- or move this to the MLflow Model Registry.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib

from taxieta import config

KEY_PACKAGES = ["pandas", "numpy", "lightgbm", "scikit-learn", "pyarrow", "duckdb"]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _deps() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in KEY_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def next_version() -> str:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in config.MODELS_DIR.glob("v[0-9]*") if p.is_dir())
    n = int(existing[-1][1:]) + 1 if existing else 1
    return f"v{n:04d}"


def save(model: Any, hist: Any, heuristic: Any, meta: dict) -> str:
    version = next_version()
    d = config.MODELS_DIR / version
    d.mkdir(parents=True)
    joblib.dump(model, d / "model.joblib")
    joblib.dump(hist, d / "hist.joblib")
    joblib.dump(heuristic, d / "heuristic.joblib")
    meta = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "dependencies": _deps(),
        "tags": {"owner": "you", "task": "eta-regression"},
        **meta,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return version


def load(version: str) -> dict:
    d = config.MODELS_DIR / version
    return {
        "version": version,
        "model": joblib.load(d / "model.joblib"),
        "hist": joblib.load(d / "hist.joblib"),
        "heuristic": joblib.load(d / "heuristic.joblib"),
        "meta": json.loads((d / "meta.json").read_text()),
    }


def champion_path() -> Path:
    return config.MODELS_DIR / "champion.json"


def set_champion(version: str) -> None:
    champion_path().write_text(json.dumps({"version": version, "since": datetime.now(UTC).isoformat()}))


def champion_version() -> str | None:
    p = champion_path()
    return json.loads(p.read_text())["version"] if p.exists() else None


def load_champion() -> dict | None:
    v = champion_version()
    return load(v) if v else None
