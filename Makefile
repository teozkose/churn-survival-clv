PY ?= python
export PYTHONPATH := src

.PHONY: data features run report notebook test lint all

data:
	$(PY) -m churn.pipeline data

features: data
	$(PY) -m churn.pipeline features

run: features
	$(PY) -m churn.pipeline run

report:
	$(PY) -m churn.pipeline report

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb --ExecutePreprocessor.timeout=600

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

all: run report notebook
