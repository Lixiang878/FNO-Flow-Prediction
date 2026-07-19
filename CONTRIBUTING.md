# Contributing

Thanks for your interest in `fno-flow-prediction`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"      # pytest + ruff
```

## Conventions

- **Offline-first.** The core (data generation, baselines, architecture forward
  passes) must run with only numpy. Anything requiring torch is lazy-imported and
  optional.
- **Tests are offline.** `pytest -q` must pass with no network and no torch.
- Keep functions small and typed; document the math, not just the API.

## Pull requests

See `.github/PULL_REQUEST_TEMPLATE.md`. Run `pytest -q` and `ruff check` before
opening a PR.
