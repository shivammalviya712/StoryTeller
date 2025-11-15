from typing import Dict, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from story_judge import StoryFeedback, StoryJudge
from storyteller import StoryTeller

# Optional: Langfuse tracing (simple callback wiring)
try:
    from langfuse.langchain import CallbackHandler
    _LANGFUSE_AVAILABLE = True
except ImportError:
    CallbackHandler = None
    _LANGFUSE_AVAILABLE = False

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

If I had 2 more hours, I would 
(1) Add a equaluation dataset against which I can text the quality of the response after each revision. It would be crucial for iteration in the architecture.
(2) Experiment with alternative agent architectures: Try multiple parallel story generations (e.g., 2-3 drafts in parallel), have the judge score each, and then only revise the best candidate ("self-competition" / tournament selection) to see if that reliably improves final quality without too much extra latency.
"""

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


class StoryState(TypedDict, total=False):
    user_request: str
    draft_story: str
    judge_feedback: StoryFeedback
    final_story: str


class StoryPipelineOutput(BaseModel):
    user_request: str = Field(..., description="Original request from the user.")
    initial_story: str = Field(..., description="Story drafted before review.")
    judge_feedback: StoryFeedback = Field(
        ..., description="Structured critique from the LLM judge."
    )
    final_story: str = Field(
        ..., description="Story returned to the user after any revisions."
    )
    revision_performed: bool = Field(
        ...,
        description="Indicates whether a revision was performed based on the judge's scores.",
    )


def build_story_graph(storyteller: StoryTeller, story_judge: StoryJudge):
    graph = StateGraph(StoryState)

    def generate_story(state: StoryState) -> Dict[str, str]:
        story = storyteller.tell_story(state["user_request"])
        return {"draft_story": story}

    def evaluate_story(state: StoryState) -> Dict[str, StoryFeedback]:
        feedback = story_judge.review(state["user_request"], state["draft_story"])
        return {"judge_feedback": feedback}

    def apply_feedback(state: StoryState) -> Dict[str, str]:
        feedback: StoryFeedback = state["judge_feedback"]

        # Decide if a revision is needed purely from scores.
        needs_revision = not StoryJudge.is_ready(feedback)

        if needs_revision:
            # Use the judge's edit_instructions directly as the revision plan.
            revision_notes = feedback.edit_instructions.strip() or (
                "Revise once to better match a calm, age-appropriate bedtime tone for "
                "children ages 5–10, with simple language and a soft, cozy ending."
            )
            revised_story = storyteller.revise_story(
                state["user_request"],
                state["draft_story"],
                revision_notes,
            )
            return {"final_story": revised_story}

        # If no revision needed, pass through the original draft.
        return {"final_story": state["draft_story"]}

    graph.add_node("generate_story", generate_story)
    graph.add_node("evaluate_story", evaluate_story)
    graph.add_node("apply_feedback", apply_feedback)

    graph.set_entry_point("generate_story")
    graph.add_edge("generate_story", "evaluate_story")
    graph.add_edge("evaluate_story", "apply_feedback")
    graph.add_edge("apply_feedback", END)

    return graph.compile()


def main():
    storyteller = StoryTeller()
    judge = StoryJudge()
    pipeline = build_story_graph(storyteller, judge)

    # Simple Langfuse tracing: attach a callback handler to the graph invocation if available.
    langfuse_handler = CallbackHandler() if _LANGFUSE_AVAILABLE else None

    print("Type 'exit' or 'quit' to end the conversation.")
    print(f"Example request: {example_requests}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        invoke_config = {}
        if langfuse_handler is not None:
            invoke_config = {
                "callbacks": [langfuse_handler],
                "tags": ["hippocratic-ai", "story_pipeline"],
            }

        # Run the LangGraph pipeline.
        state_result = pipeline.invoke({"user_request": user_input}, config=invoke_config)

        feedback: StoryFeedback = state_result["judge_feedback"]
        final_story = state_result["final_story"]
        initial_story = state_result["draft_story"]

        revision_performed = not StoryJudge.is_ready(feedback)

        response = StoryPipelineOutput(
            user_request=user_input,
            initial_story=initial_story,
            judge_feedback=feedback,
            final_story=final_story,
            revision_performed=revision_performed,
        )

        print("\nStoryteller:\n")
        print(response.final_story)

        # Judge view (scores + issues + instructions)
        print("\nJudge Review:")
        scores = response.judge_feedback.scores
        print("Scores:")
        print(f"  - Age fit:              {scores.age_fit}")
        print(f"  - Safety & sensitivity: {scores.safety_sensitivity}")
        print(f"  - Clarity & structure:  {scores.clarity_structure}")
        print(f"  - Bedtime tone:         {scores.tone_bedtime}")
        print(f"  - Engagement:           {scores.engagement_creativity}")
        print(f"  - Length fit:           {scores.length_fit}")

        status = "APPROVED ✅" if StoryJudge.is_ready(feedback) else "REVISION SUGGESTED 🔁"
        print(f"\nOverall status: {status}")

        if response.judge_feedback.issues:
            print("\nIssues:")
            for issue in response.judge_feedback.issues:
                print(f"  • {issue}")

        if response.judge_feedback.edit_instructions:
            print("\nRevision plan from judge:")
            print(f"  {response.judge_feedback.edit_instructions}")

        print(f"\nRevision performed: {response.revision_performed}\n")


if __name__ == "__main__":
    main()
