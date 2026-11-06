---
title: Blood Cell Detector
emoji: 🩸
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Detect and count WBCs, RBCs and platelets in blood smears
tags:
  - object-detection
  - yolo
  - ultralytics
  - medical-imaging
  - microscopy
---

# Blood cell detector

Drop a peripheral blood smear image to find and count its white blood cells (WBC), red blood cells (RBC) and
platelets. The model is `yolo26m-txlpbc-mine.pt`: a COCO-pretrained YOLO26m trained on
[TXL-PBC](https://github.com/lugan113/TXL-PBC_Dataset), then fine-tuned on personal **400× and 1000× microscope
photos**. Red cell size is measured automatically and each photo is analyzed in overlapping tiles at the training
cell size. It runs on ZeroGPU: each image gets a GPU for a
moment, and the confidence slider and display options then work on the CPU without running the model again.

**Cell detector model** switches between the microscope fine-tune (`yolo26m-txlpbc-mine.pt`) and the original
TXL-PBC detector (`yolo26m-txlpbc.pt`). Choose independently in the Image and Live camera tabs. Switching keeps
the detection confidence threshold; live mode clears tracks and recalibrates. Both use the same tiling pipeline.
The WBC subtype classifier is shared by both detectors, and CSV exports identify the detector used.

Detected WBCs are also classified by `resnet18-bloodmnist-wbc.pt` into basophil, eosinophil, immature granulocytes,
lymphocyte, monocyte or neutrophil. The Image tab shows subtype labels and counts, and adds subtype/confidence
columns to the table and CSV. A separate subtype threshold (initially **0.70**) marks lower-scoring predictions
as *Uncertain*. This classifier was evaluated on BloodMNIST cell crops; subtype accuracy on personal microscope
photos and the complete detection/classification pipeline have not been measured.

The *Live camera* tab detects cells in webcam or microscope-camera video as it plays. It runs the model's ONNX export
in your browser with ONNX Runtime Web, on your GPU through WebGPU where the browser supports it and on the CPU
otherwise, much more slowly. It calibrates red cell size and processes overlapping tiles; use *Recalibrate* after
changing microscope objectives. The first start downloads about 110 MB, which the browser keeps. Video stays on
your device; *Capture frame* sends one frame to the *Image* tab for the full analysis on the Space's GPU.
For live subtype labels using the local NVIDIA GPU, run `make app` from the repository.

The default confidence is approximately **0.42**, selected on personal validation tiles. The app's *About the model*
panel separates personal validation results from TXL-PBC test results, and reports count errors per tile. Whole-photo
and live-camera accuracy have not been measured independently. Training,
evaluation and this app are in [nicoaira/blood-cells-project](https://github.com/nicoaira/blood-cells-project).

This is a research demo, not a diagnostic device. The model and app are AGPL-3.0, the license of the Ultralytics
YOLO26 weights they build on. The example images come from the TXL-PBC test split (MIT License); please cite
Gan, L., Li, X. & Wang, X. *Sci Data* 12, 1694 (2025), [doi:10.1038/s41597-025-05980-z](https://doi.org/10.1038/s41597-025-05980-z).
