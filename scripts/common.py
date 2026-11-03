"""Shared paths, class metadata and helpers for the project scripts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "TXL-PBC"
CONFIG_FILE = ROOT / "config" / "train.yaml"
KERNEL_SOURCE = ROOT / "kaggle" / "train_kernel.py"
BUILD_DIR = ROOT / "build"
RUNS_DIR = ROOT / "runs"
CURRENT_RUN_FILE = RUNS_DIR / "current_run.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DASHBOARD_DIR = ROOT / "dashboard"
DASHBOARD_DATA_DIR = DASHBOARD_DIR / "data"

SPLITS = ("train", "val", "test")
CLASS_NAMES = ("WBC", "RBC", "Platelets")
EVENT_PREFIX = "@@BC@@ "  # must match kaggle/train_kernel.py


def load_config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def kaggle_api():
    """Return an authenticated KaggleApi.

    The token comes from KAGGLE_API_TOKEN in .env. It is loaded into the process
    environment before the kaggle package is imported, because importing it
    authenticates immediately.
    """
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is missing: add it to .env (see .env.example)")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def read_current_run() -> dict:
    if not CURRENT_RUN_FILE.exists():
        raise SystemExit("No Kaggle run recorded yet: push one with `make train`")
    return json.loads(CURRENT_RUN_FILE.read_text(encoding="utf-8"))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path: Path, obj, indent: int | None = 2) -> None:
    _atomic_write(path, json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def write_dashboard_data(name: str, obj, directory: Path = DASHBOARD_DATA_DIR) -> Path:
    """Write dashboard/data/<name>.js, a script that assigns window.BC_DATA[name].

    The dashboard loads its data through <script> tags instead of fetch(), so it
    also works when index.html is opened straight from disk (file://).
    """
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    path = directory / f"{name}.js"
    _atomic_write(path, f"(window.BC_DATA = window.BC_DATA || {{}})[{json.dumps(name)}] = {payload};\n")
    return path


def read_dashboard_data(name: str):
    """Inverse of write_dashboard_data; None when the file does not exist yet."""
    path = DASHBOARD_DATA_DIR / f"{name}.js"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8").split("] = ", 1)[1].rstrip().removesuffix(";"))


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Rows of (class_id, x_center, y_center, width, height), all normalized."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows


def relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()
