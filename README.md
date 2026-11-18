# Agent Harness Safety & Model Routing Solver

Profile: `agent-harness-2026-09`

Universal safety layer solver deployed on Render.
Classifies requests across all 4 policy axes:
- `requestDecision`: `allow` / `block` / `review`
- `toolDecision`: `allow` / `block` / `review` / `none`
- `route`: `primary` / `restricted` / `none`
- `trajectory`: `normal` / `watch` / `intervene`

## Endpoints
- `GET /classify` -> Configuration object for Check / Save
- `POST /classify` -> Batch evaluation endpoint
