.PHONY: setup build run test e2e reproduce train format
setup:
	uv sync --frozen
	npm ci --prefix web
build:
	npm run build --prefix web
run: build
	uv run uvicorn shirabe.api:app --host 127.0.0.1 --port 8787
test:
	uv run ruff check shirabe scripts tests
	uv run pytest -q
reproduce:
	uv run python scripts/reproduce_style.py
train:
	uv run python scripts/fetch_data.py
	uv run python scripts/train.py
e2e: build
	cd web && npm run test:e2e
format:
	uv run ruff format shirabe scripts tests
	web/node_modules/.bin/prettier --write web/src web/e2e web/*.ts web/*.json web/index.html
