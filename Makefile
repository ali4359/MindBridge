.PHONY: eval-mental-health eval-legal

PYTHON ?= .venv/bin/python

eval-mental-health:
	$(PYTHON) run_eval.py --config configs/mental_health.yaml

eval-legal:
	$(PYTHON) run_eval.py --config configs/legal.yaml
