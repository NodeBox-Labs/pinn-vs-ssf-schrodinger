.PHONY: help install test lint train smoke clean

help:
	@echo "Available targets:"
	@echo "  install    Install dependencies"
	@echo "  test       Run all tests"
	@echo "  lint       Run ruff + mypy"
	@echo "  train      Run full benchmark (all cases)"
	@echo "  smoke      30-second smoke test"
	@echo "  clean      Remove checkpoints and caches"

install:
	pip install -r requirements.txt
	pip install ruff pytest pytest-cov mypy tabulate pre-commit
	pre-commit install

test:
	python tests/test_reference_convergence.py --case a
	python tests/test_solver_against_analytic.py
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports || true

smoke:
	python src/train.py --config configs/case_a_free_gaussian.yaml \
	    --override training.iters=50 training.log_every=10

train:
	bash scripts/run_all.sh
	python src/evaluate.py --results_dir results/checkpoints \
	                        --output results/tables/master_results.csv \
	                        --readme README.md

clean:
	rm -rf results/checkpoints/*.pt results/checkpoints/*.json
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache .mypy_cache