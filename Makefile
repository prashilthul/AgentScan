.PHONY: up down seed run scan eval demo test lint

up:        ## start fake AWS + tracing stack
	docker compose up -d
down:
	docker compose down -v
seed:      ## create the fake client account (resources, IAM, trail, log groups)
	uv run python -m lab.seed
run:       ## run the real workloads (real LLM calls) -> traces + AWS-shaped logs
	uv run python -m lab.run
scan:      ## run the crawler
	uv run agentscan scan --out out/report.json
eval:      ## score the report against ground truth
	uv run agentscan eval out/report.json lab/ground_truth.json
demo: up seed run scan eval
test:
	uv run pytest -q
lint:
	uv run ruff check .
