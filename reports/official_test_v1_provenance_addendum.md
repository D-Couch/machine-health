# Official test v1 provenance clarification

The frozen protocol was committed at `a2223a5` before official test scoring. The scoring implementation itself had not yet been committed when the first report was generated; Git was at `a2223a5` with a dirty working tree. The report's `code_revision` identifies that base HEAD, not a clean evaluator snapshot.

The original `official_test_v1.json` and `.md` are preserved without changing their metrics. The evaluator and shared data/model source hashes are recorded in the accompanying JSON. No model or threshold selection used official test feedback, and no rescore was performed for this clarification. Treat this as a documented provenance deviation from the ideal of committing evaluator code before scoring.
