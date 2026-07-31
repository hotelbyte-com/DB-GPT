# Repository Agent Guide

## Responsibility

This is the HotelByte fork of the upstream DB-GPT Python/TypeScript monorepo. It owns DB-GPT packages, app/API layers, data-source connectors, agents, web UI, docs, and packaging. It is not a HotelByte Go service: do not import hotel-be build, RBAC, schema, money, or Go test conventions here.

Keep changes minimal and upstream-reviewable. Prefer a narrow adapter or focused fix over a fork-wide abstraction, dependency refresh, formatting pass, or generated-file churn.

## Read First

- `README.md`, `CONTRIBUTING.md`, `pyproject.toml`, `uv.lock`, and `Makefile` for the Python workspace and supported commands.
- The affected package's `pyproject.toml`, README, adjacent tests, and public exports under `packages/`.
- `packages/dbgpt-app/` for the runnable app/API and chat scenes, `packages/dbgpt-core/` for core agent/model contracts, `packages/dbgpt-ext/` for connectors/tools, and `packages/dbgpt-serve/` for service modules.
- `web/package.json` and `web/README.md` for Web UI work; `docs/package.json` and `docs/README.md` for Docusaurus work.
- For the HotelByte overlay, inspect recent history and the relevant paths around `component_configs.py`, `agentic_data_api.py`, `datasource_router.py`, `tool_hotelbe.py`, `conn_tdengine.py`, Redis/MinIO tools, and `scripts/seed_datasources.py` before editing.

## Fork And Upstream-Sync Boundary

- `origin` is the HotelByte fork; no upstream remote is configured in this checkout. Do not assume `origin/main` equals `eosphoros-ai/DB-GPT` upstream.
- Upstream synchronization is its own task. Identify the exact upstream base, keep sync commits separate from HotelByte behavior changes, and preserve or deliberately reapply the local overlay with focused tests.
- Do not combine an upstream merge/rebase with feature work, mass formatting, package moves, translated-doc rewrites, or a broad `uv.lock` regeneration. Report conflicts and semantic choices explicitly.
- Avoid embedding HotelByte policy in generic upstream modules when an extension/adapter boundary works. When a generic path must change, preserve upstream defaults and cover both generic and HotelByte behavior.

## Dangerous And Generated Boundaries

- Treat `.env`, `configs/*.toml`, datasource definitions, model/provider settings, and local profile files as sensitive even when examples are tracked. Never add, echo, copy into reports, or commit live keys, DSNs, tokens, or customer data; new configuration uses environment-backed placeholders.
- Agent tools can execute SQL, Python, files, network calls, Redis/MinIO actions, and external model requests. Preserve sandbox, authorization, allowlist, and confirmation boundaries; tests must not silently touch live services.
- A failed tool, unavailable datasource, or empty query is not success. Return an explicit error/gap and retain source identity; never inject plausible demo rows into runtime results.
- `uv.lock` is generated but versioned; change it only for intentional workspace/dependency changes. `.venv/`, `.venv.make/`, `dist/`, coverage output, `web/.next/`, and `web/out/` are generated and must not be hand-edited.
- `make publish`, `make publish-test`, releases, Docker pushes, migrations, and live datasource seeding are outward/destructive actions and require explicit task scope.

## Minimum Verification

- Any change: `git diff --check` and inspect the complete diff for unrelated fork churn.
- Python: run `uv run ruff check` on the changed Python paths and `uv run pytest` on the nearest affected test file/package. Use `make test` only when the broader Python suite is warranted; use `make mypy` for core typing changes.
- `make fmt` mutates the tree. `make fmt-check` also currently contains `ruff check --fix`; neither is a read-only validation step. Run them only intentionally and inspect all resulting changes.
- Web UI: from `web/`, use the CI path `yarn build`; add focused UI checks for the changed behavior.
- Docs: from `docs/`, run `npm run build` for Docusaurus content/config changes.
- Connector, model-provider, SQL/code-execution, or runtime configuration changes need focused unit tests first. Run live integration only with explicit credentials/environment scope, and report unit, integration, and live evidence separately.

## Agent Capability Contract

- Keep tool/request/result contracts typed and validated. Unknown tools, invalid arguments, parser failures, and unsupported providers must fail visibly.
- Preserve caller-selected datasource semantics through logical-to-concrete routing. Ambiguity or an unavailable source must produce a diagnosable gap, not an arbitrary fallback to another database.
- Claims that an agent, connector, provider, or sandbox supports a capability require a registered runtime path plus a test that exercises the real boundary; documentation or mocked happy paths alone are insufficient.
- Do not parse product intent or suppress errors with brittle substring checks when a typed field, parser state, registry, or configuration boundary is available.

## Code Review Rules

### Keep the fork syncable

- Flag broad refactors, formatting churn, dependency/lockfile changes, or upstream imports mixed with a local behavior fix. Safe path: isolate upstream synchronization and keep the HotelByte patch minimal with explicit compatibility tests.

### Preserve datasource identity

- Flag routing that silently substitutes a different datasource, drops caller source semantics, or converts ambiguity into a default. Safe path: resolve from typed metadata and return an explicit unresolved/ambiguous error when it cannot be proven.

### Guard tool and code execution

- Flag SQL/code/file/network/cache/object-store execution that bypasses validation, sandboxing, authorization, or intended confirmation. Safe path: validate typed arguments, enforce the narrow execution policy, and expose failures without running a fallback action.

### Prevent fabricated success

- Flag mock/demo data entering production paths, empty-success error handling, or capability claims supported only by prose. Safe path: return a structured gap/error and add a focused proof at the registration-to-execution boundary.

### Protect secrets and public contracts

- Flag new secrets in tracked config and breaking changes to package exports, APIs, event/stream shapes, or config keys without compatibility handling. Safe path: use environment references and preserve the old contract or provide a tested migration.
