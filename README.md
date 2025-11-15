# StoryTeller – Bedtime Story Agent

A small agentic system for generating and refining bedtime stories (ages 5–10) using `gpt-3.5-turbo`.  
The pipeline uses a **StoryTeller** to draft stories and a **StoryJudge** to score and optionally request a revision, wired together with **LangGraph**.

---

## 1. Setup

### Prerequisites
- Python 3.12
- An OpenAI API key

### Install dependencies

```bash
# (optional) create and activate a virtual env
# python -m venv .venv
# source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root (or export these variables):

```env
OPENAI_API_KEY=your_openai_key_here

# Optional: Langfuse tracing (if installed and configured)
# LANGFUSE_PUBLIC_KEY=...
# LANGFUSE_SECRET_KEY=...
# LANGFUSE_HOST=https://cloud.langfuse.com
```

### Run the CLI

```bash
python main.py
```

You’ll be prompted for a bedtime story request (e.g.  
`A story about a girl named Alice and her best friend Bob, who happens to be a cat.`)

The program will:
1. Generate a story.
2. Judge it using a rubric.
3. Optionally revise it if scores are below threshold.
4. Print the final story and judge scores.

---

## 2. High-Level Flow

```mermaid
flowchart LR
    U[User] -->|bedtime story request| G[Story pipeline]

    subgraph Pipeline
        direction LR

        %% 1) Generate story
        G --> GS[Node: generate_story]
        GS -->|system prompt + history + user_request| ST[StoryTeller.tell_story]
        ST -->|draft_story| EV[Node: evaluate_story]

        %% 2) Judge story
        EV -->|user_request + draft_story| J[StoryJudge.review]
        J -->|StoryFeedback (scores, issues, edit_instructions)| FB[judge_feedback]

        %% 3) Decide: ready or needs revision
        FB --> DEC{StoryJudge.is_ready\n(all scores >= threshold)}
        DEC -->|yes| FINAL_PASS[final_story = draft_story]
        DEC -->|no| AP[Node: apply_feedback]

        %% 4) Revise story if needed
        AP -->|user_request + draft_story + edit_instructions| STR[StoryTeller.revise_story]
        STR -->|revised_story| FINAL_REV[final_story]

    end

    %% Output to user
    FINAL_PASS --> OUT[Print final_story + StoryFeedback]
    FINAL_REV --> OUT

    %% Optional tracing
    G -. callbacks .-> L[Langfuse (optional)]
```

---

## 3. Components

### `StoryTeller` (`storyteller.py`)
- Wraps `ChatOpenAI(model="gpt-3.5-turbo")`.
- Loads a bedtime-story **system prompt** from `system_prompt.txt`.
- Maintains an `InMemoryChatMessageHistory` so the model can see prior turns.
- Methods:
  - `tell_story(user_request: str) -> str`  
    Generates the initial bedtime story (age-appropriate, gentle, and comforting).
  - `revise_story(user_request: str, draft_story: str, feedback: str) -> str`  
    Refines the original story using feedback (`edit_instructions`) from the judge while keeping characters, plot, and length broadly similar.

### `StoryJudge` & `StoryFeedback` (`story_judge.py`)
- Uses `ChatOpenAI` with `with_structured_output(...)` to return a **typed** `StoryFeedback`.
- `StoryFeedback` includes:
  - `scores`: integers 0–5 for  
    `age_fit`, `safety_sensitivity`, `clarity_structure`, `tone_bedtime`, `engagement_creativity`, `length_fit`
  - `issues`: short strings describing key problems (if any)
  - `edit_instructions`: a compact, actionable revision plan (≤ ~120 words)
  - `metadata`: optional echo of age, tone, target length, etc.
- Policy helper:
  - `StoryJudge.is_ready(feedback: StoryFeedback, threshold: int = 4) -> bool`  
    Returns `True` if all scores meet or exceed the threshold (default 4).

### LangGraph pipeline (`main.py`)
- Defines a `StoryState` with:
  - `user_request`, `draft_story`, `judge_feedback`, `final_story`
- Nodes:
  - `generate_story` → calls `StoryTeller.tell_story`
  - `evaluate_story` → calls `StoryJudge.review`
  - `apply_feedback` → calls `StoryJudge.is_ready`  
    - If not ready: calls `StoryTeller.revise_story` with `edit_instructions`  
    - If ready: forwards the draft unchanged
- Compiles a runnable graph and exposes it via a simple CLI loop.
- Optionally attaches a Langfuse callback handler to `pipeline.invoke(...)` for tracing.

---

## 4. Future Extensions (Agentic Improvements)

If extended beyond the initial 2-hour implementation window, the next steps would be:

- Add a small **evaluation dataset** of representative bedtime story requests with expected rubric ranges, plus a harness to automatically track how changes affect scores.
- Experiment with **parallel draft generation** (self-competition): generate multiple drafts, score them with the judge, and only revise the best candidate.
- Refine the judge’s rubric and the revision prompt into a more stable “policy improvement” loop, and deepen tracing/analytics to understand and iterate on failure modes.

---
