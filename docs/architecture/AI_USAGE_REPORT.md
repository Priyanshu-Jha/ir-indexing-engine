# AI Usage Report

## Tool and Approach
- **Tool used:** GitHub Copilot Coding Agent (LLM-assisted drafting).
- **Approach:**
  1. Read repository modules and runtime flow.
  2. Draft initial C1/C2/C3 diagrams as editable PlantUML.
  3. Review and refine diagrams against actual code relationships.
  4. Draft ADRs and align each one with observable implementation choices.
  5. Export final refined diagrams to PNG for submission readiness.

## Artifact-wise Details

### 1) C1 Diagram
- **Initial generated version contained:** user, system, and high-level external dependencies.
- **Changes made:** clarified interaction boundaries (dataset input, NLTK dependency, dual persistence targets, evaluation artifacts).
- **Why changes were necessary:** to accurately reflect code-level dependencies and avoid overly generic system-context output.

### 2) C2 Diagram (Container Diagram)
- **Initial generated version contained:** a single app container and database/filesystem.
- **Changes made:** split into concrete containers matching repository modules (orchestrator, indexing, query, evaluation, visualization, text processing, datastore adapter) and connected data flow.
- **Why changes were necessary:** assignment expects meaningful container decomposition rather than a coarse high-level sketch.

### 3) C3 Diagram (Component Diagram)
- **Initial generated version contained:** generic indexing/query components.
- **Changes made:** mapped explicit components/functions/classes (`main`, chunk helpers, indexer hierarchy, parser/evaluator, framework, datastore, text processor).
- **Why changes were necessary:** to align with real code structure and inheritance/usage relationships in this repository.

### 4) ADRs
- **Initial generated version contained:** broad architecture decisions without repository grounding.
- **Changes made:** converted to 5 concise ADRs tied directly to implemented patterns (CLI orchestration, multiprocessing chunks, dual datastore, configurable compression, TAAT/DAAT support).
- **Why changes were necessary:** decisions must be traceable to current implementation and assignment expectations.
