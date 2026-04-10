# ADR-002: Use Chunk-Based Multiprocessing for Index Construction

- **Status:** Accepted
- **Date:** 2026-04-10

## Context
Indexing a large document sample can be slow when processed serially and can exceed memory limits if handled as a single in-memory build.

## Decision
Split documents into chunks, build partial indexes in parallel worker processes, persist partial blocks, and then merge blocks into final indexes.

## Consequences
- Positive: improved throughput and better memory behavior during indexing.
- Negative: added complexity in partial block handling and merge logic.
