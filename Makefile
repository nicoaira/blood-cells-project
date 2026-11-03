PY ?= .venv/bin/python

.PHONY: help setup setup-ml dataset-report smoke upload train monitor reports dashboard predict

help:            ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:           ## create .venv with the Kaggle, monitoring and reporting dependencies
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt

setup-ml:        ## add CPU PyTorch + Ultralytics for the smoke test and local inference
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu torch torchvision
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
