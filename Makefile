.PHONY: up down seed run scan approve-all eval demo test lint

up:        ## start fake AWS + tracing stack
	docker compose up -d
down:
	docker compose down -v
seed:      ## create the fake client account (resources, IAM, trail, log groups)
	uv run python -m lab.seed
run:       ## run the real workloads (real LLM calls) -> traces + AWS-shaped logs
	uv run python -m lab.run
scan:      ## run the crawler with the sources approved so far (out/grants.json)
	uv run agentscan scan --out out/report.json
approve-all:  ## lab shortcut: play the client and approve every source at once
	for s in iam cloudtrail invocation-logs app-logs traces; do \
		uv run agentscan approve $$s --by "lab client" --yes; done
eval:      ## score the report against ground truth
	uv run agentscan eval out/report.json out/ground_truth.json
demo: up seed run   ## day 1 scan, approve everything, scan again, score
	$(MAKE) scan approve-all
	$(MAKE) scan eval
test:
	uv run pytest -q
lint:
	uv run ruff check .
