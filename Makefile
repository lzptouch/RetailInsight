PY ?= python
SCALE ?= 0.3

.PHONY: install gen etl rfm rules churn profile cluster reports all test cov lint clean

install:
	$(PY) -m pip install -e ".[dev]"

gen:
	$(PY) -m retailinsight.cli run --stage gen --scale $(SCALE)

etl:
	$(PY) -m retailinsight.cli run --stage etl

rfm:
	$(PY) -m retailinsight.cli run --stage rfm

rules:
	$(PY) -m retailinsight.cli run --stage rules

churn:
	$(PY) -m retailinsight.cli run --stage churn

profile:
	$(PY) -m retailinsight.cli run --stage profile

cluster:
	$(PY) -m retailinsight.cli run --stage cluster

reports:
	$(PY) -m retailinsight.cli run --stage reports

all:
	$(PY) -m retailinsight.cli run --stage all --scale $(SCALE)

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=retailinsight --cov-report=term-missing

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
