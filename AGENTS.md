# Machine Health — working instructions

Read PROJECT_BRIEF.md before implementation and docs/STATUS.md when it exists. The brief defines scope and evaluation. Keep this file concise and update actual setup/check commands as implementation establishes them.

- Build working end-to-end milestones. The first task is Milestone 1, including real baseline training, validation, artifact loading, and a replay dashboard.
- Make routine technical choices independently and continue through implementation and verification. Ask only when missing information materially blocks work or an action requires permission under the active environment rules.
- Inspect the actual host before selecting Python, dependencies, containers, or GPU paths. Install project dependencies in isolation; CPU baseline execution is required. Do not modify host GPU drivers or global system configuration.
- For GPU-dependent runs, if the default command sandbox cannot see a host GPU, retry the operation through the approved escalated execution path before concluding the machine has no GPU. Verify the run report's device and GPU name; keep CPU fallback and CI working.
- Preserve existing work. Use ordinary local Git commits for verified milestones when possible. Inspect existing remotes/authentication without exposing secrets. Remote creation, pushes, public releases, and paid services require user instruction; continue all independent local work if remote access is unavailable.
- Split by engine before preprocessing. Fit transforms only on training data. Features must be causal. Keep future observations, lifetime labels, and official test feedback out of training/model selection.
- Use shared preprocessing for training, dashboard, and API. Persist transforms, feature schema, model, configuration, and provenance together.
- Never fabricate results, screenshots, successful checks, dataset access, or GPU availability. Separate synthetic tests from real-data evaluation and validation metrics from official test metrics.
- Test data parsing, label semantics, engine isolation, feature causality, artifact round-trip, and the real integration boundaries. Run relevant tests and report failures or unavailable checks accurately.
- Prefer small modules, typed public interfaces, validated inputs, bounded experiments, reproducible seeds, and a simple dependency set.
- Keep datasets, large checkpoints, environments, credentials, and personal files out of Git. Verify dataset attribution/redistribution terms.
- Update docs/STATUS.md after meaningful progress with results, exact commands, limitations, and next steps. End each work session with what works, what was tested, and what remains.
