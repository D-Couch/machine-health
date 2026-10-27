# Milestone 5 task-based usability review

**Review type:** structured author self-review of the local Streamlit prototype, 2026-10-27. No maintenance or reliability practitioner participated, so this is not external user research and does not establish usability with target users.

## Scenario and tasks

The reviewer is a reliability planner looking at a simulated FD001 validation fleet at a shared benchmark cycle. Queue status is prediction-only; the held-out full-trajectory labels are available only in a separate retrospective section.

| Task | Observation | Result / follow-up |
|---|---|---|
| Find engines marked for review | The Fleet review tab leads with Review, Watch, and Routine counts and a ranked engine queue; threshold, persistence, and horizon controls are visible. | Completed. Counts are benchmark cohort counts, not staffing estimates. |
| Explain why one engine is prioritized | Selecting a queue row exposes the snapshot prediction, persistent-alert context, prediction history, sensor history through the snapshot, model name, and artifact hash. | Completed. Future practitioner review should test whether alert rationale is understandable without author explanation. |
| Choose and record a disposition | The local-only form offers inspect, monitor, defer, and dismiss, an optional rationale, and a per-engine history. A temporary-database AppTest submission displayed a success message and one persisted history event. | Completed. The log records snapshot and model/policy provenance and is not connected to CMMS or model training. |
| Compare policy scenarios | Threshold, persistence, and horizon controls update the prediction queue and a separately labeled retrospective validation scenario. | Completed. Repeated use on this small validation cohort can overfit; no policy is presented as an operational optimum. |
| Separate evidence from assumptions | Benchmark simulation scope and evidence limitations are described in the overview and methods; actual RUL is behind a retrospective evaluation expander. The intended limited-attention benefit is stated as a design goal, not a measured outcome. | Completed. No costs, staffing savings, or real-fleet impact are claimed. |

## Method and evidence

The review was a scripted task walkthrough in Streamlit `AppTest`, backed by the real local FD001 artifacts for rendering and an isolated temporary SQLite database for the disposition submission. Unit tests verify append-only events, engine filtering, saved prediction/policy/artifact context, validation-engine namespace checks, and invalid input rejection. The history schema contains no actual-RUL field. The form appears before the retrospective outcomes in the workflow.

This review was not a moderated study, accessibility audit, or visual browser review. External practitioner feedback remains outstanding; browser-level visual review belongs to Milestone 6. No official test sensors or labels were loaded or evaluated for this work.
