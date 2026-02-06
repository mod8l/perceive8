.PHONY: test test-integration test-all dev install migrate

test:
	PYTHONPATH=src poetry run pytest -v --ignore=tests/test_integration.py

test-integration:
	PYTHONPATH=src railway run poetry run pytest -v -s tests/test_integration.py -m integration

test-all:
	PYTHONPATH=src poetry run pytest -v

dev:
	lsof -ti:8000 | xargs kill -9 2>/dev/null || true; sleep 1; PYTHONPATH=src railway run .venv/bin/python -m uvicorn perceive8.main:app --reload

install:
	poetry install --with dev

migrate:
	PYTHONPATH=src railway run .venv/bin/python -m alembic upgrade head
