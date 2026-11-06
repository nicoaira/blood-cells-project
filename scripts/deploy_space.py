"""Stage the web app and deploy it to Hugging Face Spaces on ZeroGPU.

    python scripts/deploy_space.py [--repo-id USER/blood-cell-detector] [--private] [--stage-only]

Staging fills build/space with space/ (app.py, live.js, requirements.txt, README.md), the trained weights, their
ONNX export for the live camera tab (web/, run in the visitor's browser), model_info.json and, when the dataset is in
data/, a few TXL-PBC test images as examples. The fine-tuned weights and validation settings come from
config/train.yaml and results/finetune.json. The shared smear.py pipeline measures and tiles microscope photos.
`make app` runs the staged folder locally. Staging needs
requirements-ml.txt for the ONNX export, which is cached in build/onnx while the weights do not change.

Deploying creates the Space with ZeroGPU hardware if it does not exist yet, then uploads the staged folder as one
commit. It authenticates with HF_TOKEN from .env, or else with the token saved by `hf auth login`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from common import BUILD_DIR, CLASS_NAMES, DATA_DIR, MODELS_DIR, RESULTS_DIR, ROOT, load_config, write_json
import smear

os.environ.setdefault("YOLO_OFFLINE", "1")  # Ultralytics (imported for the export) stays off the network

SPACE_SOURCE = ROOT / "space"
STAGE_DIR = BUILD_DIR / "space"
SPACE_NAME = "blood-cell-detector"
HARDWARE = "zero-a10g"  # the Hub's name for ZeroGPU hardware
ONNX_CACHE = BUILD_DIR / "onnx"
# One test image per source resolution in TXL-PBC: 640x480 with platelets, 360x363 with two WBCs, 575x575, and a
# dense 256x256 field.
EXAMPLES = (
    "bac60b0b98055c6ca9ab5ec0b6e4ec3a",
    "45bc2bb9c9bd57688c652db42de77007",
    "ae15ea64f52858218a19dee8882da8f0",
    "940b907e4df95624a2ab11407a289ebb",
)


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def github_url() -> str | None:
    """The browsable URL of the origin remote, when it is on GitHub."""
    url = git("remote", "get-url", "origin") or ""
    url = url.removesuffix(".git").replace("git@github.com:", "https://github.com/")
    return url if url.startswith("https://github.com/") else None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def export_onnx(weights: Path, size: int) -> Path:
    """The ONNX export of the weights for a size x size input, from build/onnx while the weights are unchanged.

    The export is the network alone, as Ultralytics exports it: per anchor a box and three class scores, before
    non-maximum suppression, which the browser applies with Ultralytics' settings (model_info.json).
    """
    cached = ONNX_CACHE / f"{weights.stem}-{size}-{digest(weights)}.onnx"
    if not cached.exists():
        from ultralytics import YOLO

        ONNX_CACHE.mkdir(parents=True, exist_ok=True)
        work = ONNX_CACHE / weights.name  # Ultralytics writes the export next to the weights
        shutil.copy2(weights, work)
        Path(YOLO(str(work)).export(format="onnx", imgsz=size, simplify=True, device="cpu")).rename(cached)
        work.unlink()
    return cached


def model_info(cfg: dict) -> dict:
    from ultralytics.cfg import DEFAULT_CFG

    metrics = json.loads((RESULTS_DIR / "finetune.json").read_text(encoding="utf-8"))
    tuned = metrics["fine_tuned"]
    run_name = cfg["finetune"]["run_name"]
    weights = MODELS_DIR / f"{run_name}.pt"
    if (ROOT / tuned["weights"]).resolve() != weights.resolve():
        raise ValueError("Fine-tuning metrics do not match the configured web app weights")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return {
        "run_name": run_name,
        "weights": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "base_model": cfg["model"],
        "ultralytics_version": cfg["kaggle"]["ultralytics_version"],
        "class_names": list(CLASS_NAMES),
        "imgsz": smear.TILE,
        "conf_floor": cfg["evaluate"]["predict_conf"],
        "operating_conf": tuned["my_val"]["f1_optimal_conf"],
        "nms_iou": DEFAULT_CFG.iou,  # what model.predict applies; the browser repeats it
        "max_det": 1000,
        "smear": {
            "red_cell_px": smear.RED_CELL_PX,
            "tile": smear.TILE,
            "overlap": smear.OVERLAP,
            "min_field": smear.MIN_FIELD,
            "duplicate_iou": smear.DUPLICATE_IOU,
        },
        "validation": tuned["my_val"],
        "validation_counting": tuned["my_val_counts"],
        "txl_test": tuned["txl_test"],
        "repo_url": github_url(),
    }


def stage(cfg: dict, *, local: bool = False) -> Path:
    """Build an independent local bundle, or the browser-enabled Hugging Face bundle."""
    dest = BUILD_DIR / "local" if local else STAGE_DIR
    info = model_info(cfg)
    original_weights = MODELS_DIR / f"{cfg['run_name']}.pt"
    original_metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    info["default_detector"] = "microscope"
    info["detectors"] = {
        "microscope": {"label": "Microscope fine-tuned", "weights": info["weights"],
                       "run_name": info["run_name"], "weights_sha256": info["weights_sha256"]},
        "original": {"label": "Original YOLO26 (TXL-PBC)", "weights": original_weights.name,
                     "run_name": cfg["run_name"],
                     "weights_sha256": hashlib.sha256(original_weights.read_bytes()).hexdigest(),
                     "test": original_metrics["metrics"]["test"]},
    }
    subtype_metrics = json.loads((RESULTS_DIR / "wbc_classification.json").read_text(encoding="utf-8"))
    subtype_weights = ROOT / subtype_metrics["weights"]
    subtype_digest = hashlib.sha256(subtype_weights.read_bytes()).hexdigest()
    if subtype_digest != subtype_metrics["weights_sha256"]:
        raise ValueError("WBC classifier weights do not match their evaluation metadata")
    info["wbc_classifier"] = {
        "weights": subtype_weights.name, "weights_sha256": subtype_digest,
        "classes": subtype_metrics["config"]["classes"], "input_size": subtype_metrics["config"]["input_size"],
        "default_conf": 0.7, "crop_padding": 0.1,
        "test_accuracy": subtype_metrics["test"]["accuracy"], "test_macro_f1": subtype_metrics["test"]["macro_f1"],
        "test_images": sum(subtype_metrics["test"]["support"]),
    }
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(SPACE_SOURCE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "scripts" / "smear.py", dest / "smear.py")
    shutil.copy2(ROOT / "scripts" / "wbc_classifier.py", dest / "wbc_classifier.py")
    shutil.copy2(subtype_weights, dest / subtype_weights.name)
    for detector in info["detectors"].values():
        weights = MODELS_DIR / detector["weights"]
        shutil.copy2(weights, dest / weights.name)
    # Live inference uses the same 640-pixel tiles as the fine-tuning data, without the old TXL-PBC crop margin.
    # The content hash makes browsers fetch the new weights instead of reusing an older export.
    if local:
        info["live"] = {"backend": "local"}  # both tabs use the local PyTorch model; no browser export needed
    else:
        size = info["imgsz"]
        (dest / "web").mkdir()
        for detector in info["detectors"].values():
            weights = MODELS_DIR / detector["weights"]
            onnx = export_onnx(weights, size)
            detector["live"] = {"onnx": f"web/{weights.stem}-{size}-{digest(onnx)}.onnx", "input_size": size}
            shutil.copy2(onnx, dest / detector["live"]["onnx"])
        info["live"] = {"backend": "browser", **info["detectors"][info["default_detector"]]["live"]}
    write_json(dest / "model_info.json", info)
    examples = [DATA_DIR / "images" / "test" / f"{stem}.png" for stem in EXAMPLES]
    examples = [p for p in examples if p.exists()]
    (dest / "examples").mkdir()
    for path in examples:
        shutil.copy2(path, dest / "examples" / path.name)
    live = "local inference" if local else info["live"]["onnx"]
    print(f"Staged {dest.relative_to(ROOT)}: app, {info['weights']}, {live}, model_info.json, "
          f"{len(examples)} of {len(EXAMPLES)} example images" + ("" if examples else " (dataset not in data/)"))
    return dest


def deploy(repo_id: str | None, private: bool, run_name: str) -> None:
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(ROOT / ".env")
    api = HfApi()
    repo_id = repo_id or f"{api.whoami()['name']}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", space_hardware=HARDWARE, private=private,
                    exist_ok=True)
    runtime = api.get_space_runtime(repo_id)
    if HARDWARE not in (runtime.hardware, runtime.requested_hardware):
        api.request_space_hardware(repo_id, HARDWARE)
    commit = git("rev-parse", "--short", "HEAD")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=STAGE_DIR,
        commit_message=f"Deploy {run_name}" + (f" from {commit}" if commit else ""),
        delete_patterns=["examples/*", "web/*", "*.pt"],  # drop files that are no longer staged
    )
    print(f"Deployed https://huggingface.co/spaces/{repo_id} (it rebuilds now; the build takes a few minutes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", help=f"Space to deploy to (default: <your user>/{SPACE_NAME})")
    parser.add_argument("--private", action="store_true", help="create the Space as private (new Spaces only)")
    parser.add_argument("--stage-only", action="store_true", help="stage build/space without deploying")
    args = parser.parse_args()
    cfg = load_config()
    stage(cfg)
    if not args.stage_only:
        deploy(args.repo_id, args.private, cfg["finetune"]["run_name"])


if __name__ == "__main__":
    main()
