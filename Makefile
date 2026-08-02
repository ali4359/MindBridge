.PHONY: eval-mental-health eval-legal test-separation

PYTHON ?= .venv/bin/python

eval-mental-health:
	$(PYTHON) run_eval.py --config configs/mental_health.yaml

eval-legal:
	$(PYTHON) run_eval.py --config configs/legal.yaml

test-separation:
	$(PYTHON) -m pytest tests/test_domain_separation.py -v
