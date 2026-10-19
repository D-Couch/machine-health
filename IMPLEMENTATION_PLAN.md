# Milestone 1 implementation plan

1. Establish isolated Python environment, verify host/runtime and NASA source; acquire FD001 and record archive/file hashes and dataset provenance.
2. Implement validated C-MAPSS parsing, cycle-based labels, deterministic engine split, causal rolling features, sklearn constant/gradient boosting pipeline, artifact metadata and validation reports.
3. Add CLI (`prepare`, `train`, `evaluate-validation`, `replay`), Streamlit replay app, README and status documentation.
4. Add targeted tests for parsing, RUL semantics, split isolation, feature causality, artifact round-trip, and fixture integration; run tests/lint plus real-data training and app smoke checks.
