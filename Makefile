.PHONY: test dev install

test:
	PYTHONPATH=src poetry run pytest -v

dev:
	lsof -ti:8000 | xargs kill -9 2>/dev/null || true; sleep 1; PYTHONPATH=src railway run .venv/bin/python -m uvicorn perceive8.main:app --reload

install:
	poetry install --with dev
