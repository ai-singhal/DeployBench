I’ll quickly run a read-only discovery pass over your workspace and the attached spec so the 4-agent split is grounded in the actual repo constraints and execution order. Then I’ll give you copy-paste prompts for each agent with clear deliverables and handoffs.

Discover plan constraints

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench)

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/CLAUDE.md)

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/plan.md#1-1), lines 1 to 400

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/plan.md#401-401), lines 401 to 695

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/.claude)

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/.claude/worktrees)

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/.gitignore)

Searched for files matching `**/*`, 8 matches

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/.claude/worktrees/jazzy-launching-fiddle/CLAUDE.md)

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/.claude/worktrees/jazzy-launching-fiddle/plan.md#1-1), lines 1 to 80

Read [](file:///Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/DeployBench/.claude/worktrees/velvet-greeting-zephyr)

## Plan: 4-Agent Claude Code Split

TL;DR: Use one lead contract and 4 specialized agents with staged handoffs so parallel work is safe: Agent 1 creates Modal/data foundation, Agent 2 builds benchmark/workers, Agent 3 builds orchestration + business logic, Agent 4 builds Streamlit UI against a frozen response schema. This follows plan.md phases and CLAUDE.md process constraints.

**Global contract (all agents must follow)**
- Canonical orchestrator response keys: `model`, `results[]`, `recommendation`, `deploy_script`.
- Canonical result item keys: `config`, `avg_latency_ms`, `p95_latency_ms`, `fps`, `peak_memory_mb`, `mAP_50`, `est_monthly_cost`, optional `error`.
- Recommendation rule (explicit): pick lowest `est_monthly_cost` among configs with `mAP_50` drop <= 0.015 vs FP32; if none qualify, pick highest `mAP_50`.
- Required quality gates: run targeted tests + `ruff check --fix .` after Python edits.

**Copy/paste to Agent 1 (Infra + Data)**
You are Agent 1 for DeployBench. Implement only foundation and data setup from plan.md, aligned with CLAUDE.md.  
Scope:
1) Create project skeleton and Modal app/image/volume wiring.  
2) Implement one-time volume setup flow with deterministic COCO-50 subset (fixed seed), filtered annotations, and integrity checks.  
3) Add a lightweight data validation routine proving image count, annotation count, and image-id consistency.  
Out of scope: worker benchmarking internals, orchestrator logic, frontend UI.  
Deliverables:
- Runtime/dependency manifest and base app wiring.  
- Deterministic data manifest (selected image ids + counts).  
- Validation output format for downstream agents.  
Verification:
- Commands for setup and validation succeed end-to-end.  
- Output artifacts are documented for Agent 2 consumption.  
Handoff artifact:
- “Data Contract” doc with exact volume paths, file names, and expected counts.

**Copy/paste to Agent 2 (Benchmark Core + Workers)**
You are Agent 2 for DeployBench. Implement benchmark engine + 4 workers using Agent 1’s data contract and plan.md.  
Scope:
1) Build shared benchmark flow (warmup, timed loop, latency/FPS/memory, prediction capture).  
2) Implement FP32, FP16, INT8, ONNX workers with consistent return schema.  
3) Implement resilience/fallback behavior for INT8 and ONNX failures without crashing worker process.  
Out of scope: orchestration fan-out policy, cost/recommendation logic, frontend.  
Deliverables:
- Worker behavior matrix: success fields, fallback fields, error fields.  
- Deterministic metric definitions and units.  
Verification:
- Each worker independently runnable for `yolov8n/s/m`.  
- At least one test path for failure-mode payload shape.  
Handoff artifact:
- “Worker Contract” doc + sample payloads (full success and partial failure).

**Copy/paste to Agent 3 (Orchestrator + Cost + Recommendation + Deploy Script)**
You are Agent 3 for DeployBench. Implement aggregation/business logic based on Agent 2 worker contract and plan.md.  
Scope:
1) Parallel spawn/get across all workers with per-worker fault tolerance.  
2) Cost estimator integration and deterministic sorting/ranking.  
3) Recommendation object using fixed tradeoff policy (cost-min under <=1.5% mAP drop vs FP32).  
4) Deploy script generator mapped to selected config.  
Out of scope: UI rendering and layout details.  
Deliverables:
- Stable orchestrator response schema for frontend.  
- Error normalization strategy when baseline FP32 fails or any config is missing.  
Verification:
- End-to-end orchestrator run returns valid JSON for success and degraded cases.  
- Recommendation logic unit-tested for tie/edge cases.  
Handoff artifact:
- “API Contract” doc + 2 frozen fixtures (all-success, partial-failure).

**Copy/paste to Agent 4 (Frontend + Integration)**
You are Agent 4 for DeployBench. Implement Streamlit UX from plan.md using Agent 3 frozen API fixtures first, then live endpoint wiring.  
Scope:
1) Build MVP UI exactly as specified (model select, daily requests, run button, results cards, two charts, recommendation, deploy script block).  
2) Start with mock fixture integration, then switch to real orchestrator endpoint.  
3) Add graceful UI states for per-config failure rows.  
Out of scope: backend model execution changes and recommendation algorithm changes.  
Deliverables:
- Single-file Streamlit frontend with strict schema mapping table.  
- No extra features beyond scope boundaries in plan.md.  
Verification:
- UI renders correctly with both fixtures and real API response.  
- Handles 3-result and 4-result scenarios without crashing.  
Handoff artifact:
- “Frontend Mapping” doc listing each widget and exact API key dependencies.

**Execution order**
- Start Agent 1 immediately.  
- Start Agent 4 in parallel with mock fixtures once Agent 3 schema draft is available.  
- Start Agent 2 after Agent 1 data contract is frozen.  
- Start Agent 3 after Agent 2 worker contract is frozen.