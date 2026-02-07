.PHONY: test test-integration test-all dev install migrate chat-ui-install chat-ui-run run run-all

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

# Chat UI
chat-ui-install:
	cd services/chat-ui && poetry install

chat-ui-run:
	-lsof -ti:8080 | xargs kill -9 2>/dev/null || true
	cd services/chat-ui && poetry run chainlit run app.py --port 8080

run:
	-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	PYTHONPATH=src poetry run uvicorn perceive8.main:app --reload --log-level info

benchmark:
	PYTHONPATH=src poetry run python scripts/benchmark_upload.py $(FILE)

# Run both services
run-all:
	-lsof -ti:8000 -ti:8080 | xargs kill -9 2>/dev/null || true
	$(MAKE) run & $(MAKE) chat-ui-run
