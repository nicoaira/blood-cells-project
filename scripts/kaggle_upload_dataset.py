"""Upload data/TXL-PBC to Kaggle as a private dataset, or as a new version of it.

The dataset is staged under build/kaggle_dataset/ as a single zip plus
dataset-metadata.json. The training kernel finds data.yaml under /kaggle/input
whether Kaggle extracts the archive or keeps it as a zip.

    python scripts/kaggle_upload_dataset.py [--notes "why this version"]
"""

from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path

from common import BUILD_DIR, DATA_DIR, kaggle_api, load_config, write_json

READY_TIMEOUT_S = 900


def stage_dataset(ref: str, title: str) -> Path:
    stage = BUILD_DIR / "kaggle_dataset"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    archive = stage / "TXL-PBC.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
        for path in sorted(DATA_DIR.rglob("*")):
            if path.is_file() and not path.name.endswith(".cache"):
                zf.write(path, path.relative_to(DATA_DIR.parent).as_posix())
    write_json(stage / "dataset-metadata.json", {
        "title": title,
        "id": ref,
        "subtitle": "Peripheral blood smear images with YOLO boxes for WBC, RBC and platelets",
        "description": "Private training copy of the TXL-PBC dataset (1,260 images, 18,143 cells) "
                       "used by the blood-cells YOLO26 project. Classes: 0 WBC, 1 RBC, 2 Platelets.",
        "licenses": [{"name": "unknown"}],
    })
    print(f"Staged {archive.name} ({archive.stat().st_size / 1e6:.1f} MB)")
    return stage


def dataset_exists(api, ref: str) -> bool:
    try:
        api.dataset_status(ref)
        return True
    except Exception:  # the API answers 404 for datasets that do not exist yet
        return False


def wait_until_ready(api, ref: str) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            status = str(api.dataset_status(ref)).lower()
        except Exception as exc:  # a just-created dataset can answer 403/404 for a few seconds
            status = f"not visible yet ({type(exc).__name__})"
        print(f"  dataset status: {status}")
        if status == "ready":
            return
        time.sleep(15)
    raise SystemExit(f"{ref} was not ready after {READY_TIMEOUT_S} s; check https://www.kaggle.com/datasets/{ref}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--notes", default="Update dataset", help="version notes when the dataset already exists")
    args = parser.parse_args()

    cfg = load_config()["kaggle"]
    api = kaggle_api()
    ref = f"{api.get_config_value('username')}/{cfg['dataset_slug']}"
    stage = stage_dataset(ref, cfg["dataset_title"])
    if dataset_exists(api, ref):
        print(f"Creating a new version of {ref}")
        result = api.dataset_create_version(str(stage), version_notes=args.notes, quiet=False, dir_mode="skip")
    else:
        print(f"Creating private dataset {ref}")
        result = api.dataset_create_new(str(stage), public=False, quiet=False, dir_mode="skip")
    if getattr(result, "error", None):
        raise SystemExit(f"Kaggle rejected the upload: {result.error}")
    wait_until_ready(api, ref)
    print(f"Ready: https://www.kaggle.com/datasets/{ref}")


if __name__ == "__main__":
    main()
