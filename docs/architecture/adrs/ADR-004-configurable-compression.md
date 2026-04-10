# ADR-004: Make Posting Compression Configurable via Zlib

- **Status:** Accepted
- **Date:** 2026-04-10

## Context
Storage footprint and query performance trade-offs are a core part of the experimental evaluation.

## Decision
Allow optional zlib compression of serialized posting-list payloads in both file and database persistence modes.

## Consequences
- Positive: reduced storage in many configurations and explicit A/B comparison path.
- Negative: CPU overhead for compress/decompress during save/load and query-time data fetch.
