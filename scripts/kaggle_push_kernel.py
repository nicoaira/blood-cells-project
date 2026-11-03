"""Push the training kernel to Kaggle and start a GPU run.

Renders build/kaggle_kernel/ (kaggle/train_kernel.py with config/train.yaml
embedded, plus kernel-metadata.json), pushes it as a private script kernel on a
T4 GPU with internet access, and records the run in runs/current_run.json so
scripts/monitor.py knows what to follow.

    python scripts/kaggle_push_kernel.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone

from common import BUILD_DIR, CURRENT_RUN_FILE, KERNEL_SOURCE, kaggle_api, load_config, write_json

PLACEHOLDER = '"__EMBEDDED_CONFIG__"'


def render_kernel(cfg: dict, kernel_ref: str, dataset_ref: str):
    stage = BUILD_DIR / "kaggle_kernel"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    source = KERNEL_SOURCE.read_text(encoding="utf-8")
    if source.count(PLACEHOLDER) != 1:
        raise SystemExit(f"{KERNEL_SOURCE} must contain the {PLACEHOLDER} placeholder exactly once")
    (stage / "train_kernel.py").write_text(source.replace(PLACEHOLDER, repr(json.dumps(cfg))), encoding="utf-8")
    k = cfg["kaggle"]
    write_json(stage / "kernel-metadata.json", {
        "id": kernel_ref,
        "title": k["kernel_title"],
        "code_file": "train_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,  # pip install ultralytics + pretrained weights download
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": k["accelerator"],
    })
    return stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="render build/kaggle_kernel/ without pushing")
    args = parser.parse_args()

    cfg = load_config()
    api = kaggle_api()
    owner = api.get_config_value("username")
    kernel_ref = f"{owner}/{cfg['kaggle']['kernel_slug']}"
    dataset_ref = f"{owner}/{cfg['kaggle']['dataset_slug']}"
    stage = render_kernel(cfg, kernel_ref, dataset_ref)
    print(f"Rendered {stage}")
    if args.dry_run:
        return

    response = api.kernels_push(str(stage), acc=cfg["kaggle"]["accelerator"])
    if response.error:
        raise SystemExit(f"Kaggle rejected the push: {response.error}")
    for field in ("invalid_dataset_sources", "invalid_tags"):
        if getattr(response, field, None):
            print(f"Warning: {field} = {getattr(response, field)}")
    run = {
        "kernel": kernel_ref,
        "version": response.version_number,
        "url": response.url,
        "dataset": dataset_ref,
        "accelerator": cfg["kaggle"]["accelerator"],
        "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": cfg["run_name"],
    }
    write_json(CURRENT_RUN_FILE, run)
    print(f"Pushed {kernel_ref} version {response.version_number}: {response.url}")
    print("Follow it with: make monitor")


if __name__ == "__main__":
    main()
