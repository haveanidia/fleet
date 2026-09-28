.PHONY: install test lint build-bank build-semantic validate benchmark native-build native-validate native-release serve

install:
	pip install -e ".[server,dev]"

test:
	pytest -q

lint:
	ruff check src tests

build-bank:
	fleet build \
		--config configs/experts.json \
		--output artifacts/fleet-bank

build-semantic:
	fleet semantic-build \
		--bank artifacts/fleet-bank \
		--anchors configs/semantic_anchors.json

validate:
	fleet validate artifacts/fleet-bank

benchmark:
	fleet benchmark \
		--bank artifacts/fleet-bank \
		--output artifacts/fleet-bank/benchmark_results.json

native-build:
	fleet native-build \
		--bank artifacts/fleet-bank \
		--output artifacts/Fleet-Semantic-4B \
		--dtype float16 \
		--max-shard-size 4GB

native-validate:
	fleet native-validate \
		--model artifacts/Fleet-Semantic-4B

native-release:
	fleet native-release \
		--bank artifacts/fleet-bank \
		--output artifacts/Fleet-Semantic-4B \
		--repo noumenon-labs/Fleet-Semantic-4B

serve:
	fleet serve \
		--router semantic \
		--bank artifacts/fleet-bank \
		--host 0.0.0.0 \
		--port 8000
