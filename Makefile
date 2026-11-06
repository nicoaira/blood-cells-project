PY ?= .venv/bin/python
LABEL_PORT ?= 8089
APP_PORT ?= 7860

.PHONY: help setup setup-ml setup-gpu dataset-report smoke upload train monitor reports dashboard predict app app-gpu app-cpu deploy-app \
	label-prepare label label-export finetune predict-smear predict-video

.PHONY: train-wbc predict-wbc

help:            ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:           ## create .venv with the Kaggle, monitoring and reporting dependencies
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt

setup-ml:        ## add CPU PyTorch, Ultralytics and Gradio for the smoke test, local inference and the web app
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu torch torchvision
	uv pip install --python $(PY) -r requirements-ml.txt

setup-gpu:       ## the same with CUDA 12.6 PyTorch, to train and predict on a local NVIDIA GPU (driver 525 or newer)
	uv pip install --python $(PY) --reinstall-package torch --reinstall-package torchvision \
		--index-url https://download.pytorch.org/whl/cu126 torch torchvision
	uv pip install --python $(PY) -r requirements-ml.txt

dataset-report:  ## dataset statistics, quality checks, figures and dashboard data
	$(PY) scripts/analyze_dataset.py

smoke:           ## 2-epoch CPU run of the whole Kaggle job, to test changes before spending GPU quota
	$(PY) kaggle/train_kernel.py --config config/train.yaml --data-root data/TXL-PBC --out smoke --smoke

upload:          ## upload data/TXL-PBC to Kaggle as a private dataset (or a new version of it)
	$(PY) scripts/kaggle_upload_dataset.py

train:           ## push the training kernel to Kaggle (GPU T4) and start a run
	$(PY) scripts/kaggle_push_kernel.py

monitor:         ## follow the run live in the dashboard, then download outputs and build the reports
	$(PY) scripts/monitor.py --open

reports:         ## rebuild training and evaluation reports from the downloaded outputs
	$(PY) scripts/build_reports.py

dashboard:       ## open the dashboard in the default browser
	$(PY) -m webbrowser -t "file://$(CURDIR)/dashboard/index.html"

predict:         ## detect and count cells in your images: make predict SOURCE=path/to/images
	$(PY) scripts/predict.py $(SOURCE)

app: app-gpu     ## run the web app locally on your NVIDIA GPU at http://127.0.0.1:7860

app-gpu:         ## both images and live camera use CUDA; fails if CUDA is unavailable
	$(PY) scripts/run_app.py --device cuda --port $(APP_PORT)

app-cpu:         ## explicitly run the local app on the CPU
	$(PY) scripts/run_app.py --device cpu --port $(APP_PORT)

deploy-app:      ## deploy the web app to Hugging Face Spaces with ZeroGPU (needs `hf auth login` or HF_TOKEN in .env)
	$(PY) scripts/deploy_space.py

label-prepare:   ## cut your pictures (data/Blood_smear_pictures_mine) into tiles pre-labeled by the model
	$(PY) scripts/labeling.py prepare

label:           ## correct the pre-labels in Label Studio at http://localhost:8089 (installed with uv on first use)
	@command -v label-studio >/dev/null || test -x $(HOME)/.local/bin/label-studio \
		|| uv tool install --python 3.12 label-studio
	$(PY) scripts/labeling.py serve --open --port $(LABEL_PORT)

label-export:    ## write the corrected labels as a YOLO dataset in data/UIC_microscope_photos (while `make label` runs)
	$(PY) scripts/labeling.py export --port $(LABEL_PORT)

finetune:        ## fine-tune the model on TXL-PBC + your labeled tiles on the local GPU, and compare it with the base
	$(PY) scripts/finetune.py

predict-smear:   ## count cells in whole microscope photos at any magnification (HEIC too): make predict-smear SOURCE=...
	$(PY) scripts/predict.py --smear $(SOURCE)

predict-video:   ## detect cells in a microscope video, frame by frame, and save it with the boxes: make predict-video SOURCE=...
	$(PY) scripts/predict_video.py $(SOURCE)

train-wbc:       ## train the six-class WBC subtype model on the local NVIDIA GPU
	$(PY) scripts/train_wbc.py

predict-wbc:     ## classify a cropped WBC: make predict-wbc SOURCE=path/to/cell.png
	$(PY) scripts/wbc_classifier.py $(SOURCE)
