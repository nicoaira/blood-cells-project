"""Build the WBC classifier evaluation report from results/wbc_classification.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import REPORTS_DIR, RESULTS_DIR, ROOT
from plotting import ACCENT, INK, MUTED, apply_style, plt, save


def build_report(result_path: Path = RESULTS_DIR / "wbc_classification.json") -> Path:
    result = json.loads(result_path.read_text())
    cfg, test, val = result["config"], result["test"], result["validation"]
    names = cfg["classes"]
    labels = [name.replace("_", " ").capitalize() for name in names]
    apply_style()
    matrix = np.asarray(test["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6.8), layout="constrained")
    display_labels = [s.replace("Immature granulocytes", "Immature\ngranulocytes") for s in labels]
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(len(names)), display_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), display_labels)
    ax.set(xlabel="Predicted subtype", ylabel="True subtype",
           title=f"WBC classification · cleaned test split (n={matrix.sum():,})")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]}\n{normalized[i, j]:.1%}", ha="center", va="center",
                    color="white" if normalized[i, j] > 0.5 else INK, fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.75, label="Fraction of true-class images")
    confusion_plot = save(fig, "wbc_confusion")
    history = result["history"]
    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    epochs = [r["epoch"] for r in history]
    for key, label, color, style in (
        ("train_accuracy", "Train accuracy (augmented)", MUTED, "--"),
        ("val_accuracy", "Validation accuracy", "#2a78d6", "-"),
        ("val_macro_f1", "Validation macro-F1", ACCENT, "-"),
    ):
        ax.plot(epochs, [100 * r[key] for r in history], label=label, color=color, linestyle=style)
    ax.axvline(result["best_epoch"], color=INK, linewidth=1, linestyle=":", label=f"Selected epoch {result['best_epoch']}")
    ax.set(xlabel="Epoch", ylabel="Percent", title="Training and validation · test data excluded from selection")
    ax.set_ylim(max(0, min(100 * r["train_accuracy"] for r in history) - 1), 100.5)
    ax.legend(loc="lower right")
    training_plot = save(fig, "wbc_training")
    rows = "\n".join(
        f"| {name} | {test['support'][i]} | {test['precision'][i]:.2%} | {test['recall'][i]:.2%} | {test['f1'][i]:.2%} |"
        for i, name in enumerate(labels))
    counts = result["dataset_audit"]["clean_counts"]
    sizes = {s: sum(c.values()) for s, c in counts.items()}
    audit = result["dataset_audit"]
    confusion_pairs = [(matrix[i, j], labels[i], labels[j]) for i in range(len(names))
                       for j in range(len(names)) if i != j and matrix[i, j]]
    confusion_pairs.sort(reverse=True)
    errors = int(matrix.sum() - np.trace(matrix))
    mixups = "; ".join(f"{a} → {b}: {n}" for n, a, b in confusion_pairs[:4])
    text = f"""# WBC subtype classifier

Trained locally on **{result['environment']['gpu']}**, using the supplied `data/bloodmnist` images.
The deliverable is [`{result['weights']}`](../{result['weights']}): a six-class ResNet-18 initialized with
Torchvision's ImageNet-1K V1 weights. This is a second-stage classifier for one cropped WBC at a time.

## Held-out results

The selected checkpoint (epoch **{result['best_epoch']}** of {result['epochs_completed']} completed epochs)
classifies **{int(np.trace(matrix)):,}/{int(matrix.sum()):,}** cleaned test crops correctly:
**accuracy {test['accuracy']:.2%}**, **macro-F1 {test['macro_f1']:.2%}**,
balanced accuracy {test['balanced_accuracy']:.2%}. Validation accuracy is {val['accuracy']:.2%},
with macro-F1 {val['macro_f1']:.2%}. The test split was evaluated after checkpoint selection.

| Subtype | Test images | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
{rows}

There are {errors} test errors. Largest off-diagonal counts: {mixups or 'none'}.

![Test confusion matrix]({confusion_plot})

## Data and leakage checks

The six WBC categories are basophil, eosinophil, immature granulocytes, lymphocyte, monocyte and neutrophil.
Erythroblast and platelet images are excluded because they are not WBC subtypes. The immature granulocyte
label is a group, not a separate prediction for each maturation stage.

The source's train/validation/test split is retained. Pixel-hash auditing found {audit['duplicate_groups']}
exact duplicate groups, {audit['cross_split_duplicate_groups']} spanning splits, and
{audit['conflicting_label_groups']} group with conflicting labels. All copies of the conflicting-label image
are quarantined. Other duplicate groups contribute one copy, preferring test over validation over training.
This excludes {len(audit['excluded'])} entries from the training/evaluation manifests. Source files are unchanged.

The resulting splits have **{sizes['train']:,} train / {sizes['val']:,} validation / {sizes['test']:,} test** images.
No exact pixel hash occurs in multiple cleaned splits. These are cleaned, WBC-only results and are not the
standard eight-class BloodMNIST benchmark. Patient-level separation and near-duplicate overlap are unknown.

The full audit, exclusions, per-file hashes and manifests are under
`runs/wbc/{cfg['run_name']}/`. Manifest SHA-256: `{audit['manifest_sha256']}`.

## Training and reproduction

Native images are RGB **64×64**. They are resized bilinearly to **{cfg['input_size']}×{cfg['input_size']}**
and normalized with ImageNet mean/std; resizing does not restore missing detail. All network layers are trained.
Training augmentation uses horizontal/vertical flips, rotations up to 180°, and mild brightness/contrast/color
variation. Validation and test use resizing and normalization only, without test-time augmentation.

Loss is class-weighted cross-entropy using inverse training frequencies, with label smoothing
{cfg['train']['label_smoothing']}. AdamW uses backbone learning rate {cfg['train']['backbone_lr']},
head learning rate {cfg['train']['head_lr']}, weight decay {cfg['train']['weight_decay']},
{cfg['train']['warmup_epochs']} warm-up epochs and cosine decay. Batch size is {cfg['train']['batch_size']};
early-stopping patience is {cfg['train']['patience']}. The checkpoint maximizes **validation macro-F1**.
Seed: {cfg['train']['seed']}. PyTorch {result['environment']['torch']},
Torchvision {result['environment']['torchvision']}, CUDA {result['environment']['cuda']}.
Training uses mixed precision; final validation, test predictions and the inference loader use FP32.
This invocation took {result['duration_seconds'] / 60:.1f} minutes for training and final evaluation.

![Training and validation]({training_plot})

```bash
make train-wbc
# Resume an interrupted run:
.venv/bin/python scripts/train_wbc.py --resume
# Classify a crop containing a single WBC:
make predict-wbc SOURCE=path/to/wbc_crop.png
```

Training config: [`config/wbc.yaml`](../config/wbc.yaml).
Machine-readable metrics: [`results/wbc_classification.json`](../results/wbc_classification.json).
Per-image test probabilities, epoch history, and restart checkpoints are in `runs/wbc/{cfg['run_name']}/`.
Weights SHA-256: `{result['weights_sha256']}`.

## Use with microscope photos

First detect a WBC with the existing YOLO model, then classify a crop containing that WBC. This model does not
find or count cells in a whole photograph. It has no unknown or non-WBC class, so an unsuitable crop can be forced
into one of the six labels. Softmax scores have not been calibrated as clinical probabilities.

The test figures describe BloodMNIST crops. **Subtype accuracy on personal microscope photos at 400× or 1000×
has not been measured.** The different staining, camera, crop framing and native image resolution can change
performance. The web app applies the classifier to crops from its three-class detector and reports detection
and subtype scores separately. These crop-level test figures do not validate that full pipeline. This is a
research model and is not validated for diagnosis.
"""
    path = REPORTS_DIR / "wbc_classification_report.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path


if __name__ == "__main__":
    print(build_report().relative_to(ROOT))
