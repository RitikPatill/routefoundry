# Contributing

Thank you for helping make model routing easier to audit.

## Before opening code

For a feature, start with the user problem and a small acceptance test. For a benchmark
change, describe the data provenance, grader, split, and how the claim will be reproduced.
Never attach private prompts, unredacted responses, credentials, or restricted datasets.

## Development

```bash
git clone https://github.com/RitikPatill/routefoundry.git
cd routefoundry
uv sync --extra dev
uv run pre-commit install
uv run routefoundry demo --output out/demo
```

Before a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/routefoundry
uv run pytest --cov
uv run python scripts/secret_scan.py
```

Prefer small pull requests. Add tests for behavior, preserve deterministic results, and
explain any schema change. Public APIs and artifacts use semantic versioning; schema
changes require a migration note.

## Benchmark integrity

- Synthetic data must be labelled synthetic.
- Do not tune on the held-out test split.
- Keep hindsight oracles visibly labelled non-deployable.
- Include simple baselines and negative results.
- Do not imply that backend-non-resident timing controls the OS filesystem cache.
- Do not add a README number without a committed reproduction path.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

