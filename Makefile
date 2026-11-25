.PHONY: up down seed run scan approve-all eval web serve demo test lint

up:        ## start Floci (fake AWS), OTel collector + Jaeger, Postgres
	docker compose up -d --wait
down:
	docker compose down -v
seed:      ## create the fake client account (resources, IAM, trail, log groups)
	uv run python -m lab.seed
run:       ## run the real workloads (real LLM calls) -> traces + AWS-shaped logs
	uv run python -m lab.run
scan:      ## crawl with the sources approved so far (out/grants.json); store scan + audit trail
	uv run agentscan scan --out out/report.json
approve-all:  ## lab shortcut: play the client and approve every source at once
	for s in iam cloudtrail invocation-logs app-logs traces; do \
		uv run agentscan approve $$s --by "lab client" --yes; done
eval:      ## score the report against ground truth
	uv run agentscan eval out/report.json out/ground_truth.json
web:       ## build the dashboard (web/out), served by `make serve`
	cd web && pnpm install --frozen-lockfile && pnpm build
serve:     ## dashboard + API on http://127.0.0.1:8000
	uv run agentscan serve
demo: up seed run   ## day 1 scan, approve everything, scan again, score, build the dashboard
	$(MAKE) scan approve-all
	$(MAKE) scan eval web
	@echo "now run: make serve"
test:
	uv run pytest -q
lint:
	uv run ruff check . && uv run ruff format --check .
	cd web && pnpm lint
