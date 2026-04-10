# ADR-001: Use a CLI-Orchestrated Modular Architecture

- **Status:** Accepted
- **Date:** 2026-04-10

## Context
The project runs large experimental sweeps across indexing, datastore, compression, optimization, and query-processing configurations. The orchestration needs to be transparent and easy to reproduce for coursework experiments.

## Decision
Keep a command-line orchestrator (`main.py`) as the control layer and delegate domain behavior to dedicated modules (`indexer.py`, `query.py`, `datastore.py`, `evaluate.py`, `core.py`).

## Consequences
- Positive: explicit run flow, simple reproducibility, and easy module-level evolution.
- Negative: no long-running service endpoint for online querying.
