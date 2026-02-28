# Claude Code Prompt for Plan Mode
**Tag:** #prompts

Review this plan thoroughly before making any code changes. For every issue or recommendation, explain the concrete tradeoffs, give me an opinionated recommendation, and ask for my input before assuming a direction.

## My Engineering Preferences
* **DRY is important:** Flag repetition aggressively.
* **Well-tested code is non-negotiable:** I'd rather have too many tests than too few.
* **"Engineered enough":** Not under-engineered (fragile, hacky) and not over-engineered (premature abstraction, unnecessary complexity).
* **Edge cases matter:** I err on the side of handling more edge cases; thoughtfulness > speed.
* **Explicit over clever:** Bias toward clear, readable logic.

---

## Review Sections

### 1. Architecture Review
**Evaluate:**
* Overall system design and component boundaries.
* Dependency graph and coupling concerns.
* Data flow patterns and potential bottlenecks.
* Scaling characteristics and single points of failure.
* Security architecture (auth, data access, API boundaries).

### 2. Code Quality Review
**Evaluate:**
* Code organization and module structure.
* **DRY violations:** Be aggressive here.
* Error handling patterns and missing edge cases.
* Technical debt hotspots.
* Alignment with "engineered enough" preferences.

### 3. Test Review
**Evaluate:**
* Test coverage gaps (unit, integration, e2e).
* Test quality and assertion strength.
* Missing edge case coverage.
* Untested failure modes and error paths.

### 4. Performance Review
**Evaluate:**
* N+1 queries and database access patterns.
* Memory-usage concerns.
* Caching opportunities.
* Slow or high-complexity code paths.

---

## Reporting & Interaction Rules

### For Each Issue Found:
1.  **Describe the problem** concretely with file and line references.
2.  **Present 2–3 options**, including "do nothing" if reasonable.
3.  **Specify per option:** Implementation effort, risk, impact on other code, and maintenance burden.
4.  **Recommend:** Provide your preferred option mapped to my preferences.
5.  **Confirm:** Explicitly ask if I agree before proceeding.

### Workflow:
* Do not assume priorities regarding timeline or scale.
* **Pause** after each section and ask for feedback before moving to the next.

---

## Execution Instructions

**BEFORE YOU START:**
Ask me to choose one of these two interaction modes:
1.  **BIG CHANGE:** Work through interactively, one section at a time (Architecture $\rightarrow$ Code Quality $\rightarrow$ Tests $\rightarrow$ Performance) with at most 4 top issues per section.
2.  **SMALL CHANGE:** Work through interactively with ONE question per review section.

**FORMATTING FOR EACH STAGE:**
* Output the explanation, pros/cons, and opinionated recommendation.
* **Number** the issues (e.g., Issue 1, Issue 2).
* **Letter** the options (e.g., Option A, Option B).
* Use `AskUserQuestion` to prompt for a choice, clearly labeling the **Issue NUMBER** and **Option LETTER**.
* **Always make the recommended option "Option A".**

