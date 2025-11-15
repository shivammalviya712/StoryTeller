
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
        J -->|StoryFeedback scores issues edit_instructions| FB[judge_feedback]

        %% 3) Decide: ready or needs revision
        FB --> DEC{is_ready scores above threshold}

```