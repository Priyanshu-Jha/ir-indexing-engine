# ADR-003: Support Both Pickle and PostgreSQL Datastores

- **Status:** Accepted
- **Date:** 2026-04-10

## Context
The assignment requires comparison of persistence choices and their impact on performance and footprint.

## Decision
Support two datastore modes:
- Pickle files for simple local persistence.
- PostgreSQL for relational storage and queryable postings.

## Consequences
- Positive: enables required experimental comparison and trade-off analysis.
- Negative: implementation and maintenance overhead for two persistence paths.
