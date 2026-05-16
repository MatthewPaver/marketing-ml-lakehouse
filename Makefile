BOOTSTRAP_PYTHON ?= python3.11
PY=python
PIP=pip
VENV=.venv
PYTHON=$(VENV)/bin/python
PIPBIN=$(VENV)/bin/pip

.PHONY: venv install run dashboard train-clf test docker-build docker-run-pipeline docker-run-dashboard

venv:
	$(BOOTSTRAP_PYTHON) -m venv $(VENV)
	$(PIPBIN) install --upgrade pip

install: venv
	$(PIPBIN) install -r requirements.txt

run:
	$(PYTHON) -m lakehouse.run_all

train-clf:
	$(PYTHON) -m lakehouse.ml.train_underpacing

dashboard:
	PYTHONPATH=$(PWD) $(VENV)/bin/streamlit run lakehouse/dashboard/app.py

test: install
	$(PYTHON) -m pytest marketing-ml/tests -q

# Docker

docker-build:
	docker build -t local-lakehouse -f lakehouse/Dockerfile .

docker-run-pipeline:
	docker run --rm -v "$(PWD)":/app local-lakehouse bash -lc "python -m lakehouse.run_all"

docker-run-dashboard:
	docker run --rm -p 8501:8501 -v "$(PWD)":/app local-lakehouse bash -lc "streamlit run lakehouse/dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
