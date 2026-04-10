# ADR-005: Provide TAAT and DAAT Query Processing Modes

- **Status:** Accepted
- **Date:** 2026-04-10

## Context
The assignment asks for multiple query strategies and comparative benchmarking.

## Decision
Implement:
- **TAAT** for boolean/phrase evaluation using parser output and set operations.
- **DAAT** for ranked retrieval with top-k scoring.

## Consequences
- Positive: supports required comparisons and richer experimentation.
- Negative: two evaluation paths increase testing and reasoning complexity.
