import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, conint

load_dotenv()


# ---------------------------
# Structured output schema
# ---------------------------

class Scores(BaseModel):
    """0–5 rubric scores (integers only)."""
    age_fit: conint(ge=0, le=5)
    safety_sensitivity: conint(ge=0, le=5)
    clarity_structure: conint(ge=0, le=5)
    tone_bedtime: conint(ge=0, le=5)
    engagement_creativity: conint(ge=0, le=5)
    length_fit: conint(ge=0, le=5)


class StoryFeedback(BaseModel):
    """
    Minimal judge output:
    - scores           : rubric for gating (caller can decide thresholds)
    - issues           : short problem statements, if any
    - edit_instructions: ONE compact revision plan for a single pass
    - metadata         : optional echo (age, tone, length, etc.)
    """
    scores: Scores
    issues: List[str] = Field(
        default_factory=list,
        description="Short problem statements; can be empty if the story is ready as-is."
    )
    edit_instructions: str = Field(
        "",
        description=(
            "≤120 words of actionable guidance for ONE revision pass. "
            "If the story is already excellent, keep this very brief or note that only tiny tweaks are needed."
        ),
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional echo: {'age': int, 'tone': str, 'target_length': int, 'word_count': int}"
    )


# ---------------------------
# Judge implementation
# ---------------------------

class StoryJudge:
    """LLM-powered reviewer that critiques bedtime stories for children ages 5-10."""

    def __init__(
        self,
        *,
        max_tokens: int = 600,      # Judge should be concise
        temperature: float = 0.1,   # Low randomness for consistent gating
        model: str = "gpt-3.5-turbo",
    ):
        self._system_prompt = (
            "You are a careful children's literature JUDGE for BEDTIME stories (ages 5-10).\n"
            "Evaluate a single story and produce:\n"
            "- rubric SCORES (0-5 integers) for: age_fit, safety_sensitivity, clarity_structure,\n"
            "  tone_bedtime, engagement_creativity, length_fit;\n"
            "- a list of concise ISSUES (if any);\n"
            "- a single EDIT_INSTRUCTIONS block with a compact revision plan.\n"
            "\n"
            "ENFORCE THESE CONSTRAINTS:\n"
            "- Age fit (5-10): simple vocabulary; mostly short sentences (~5-15 words); "
            "concrete imagery; avoid complex metaphors, sarcasm, or adult topics.\n"
            "- Safety & sensitivity: no fear/violence/weaponry/gore/nightmares; no spooky "
            "entities (monster/ghost/haunted); no stereotypes; inclusive language; no "
            "medical/health claims or prescriptive advice.\n"
            "- Structure & clarity: Beginning → gentle problem → kind resolution → soft landing; "
            "easy to follow; no cliffhangers.\n"
            "- Bedtime tone: calming, decelerating energy; soothing final paragraph (stars, "
            "moon, breeze, cozy). Avoid overstimulation.\n"
            "- Engagement & creativity: light whimsy; mild sensory details; charming but gentle.\n"
            "- Length fit: near target length (default 500 words) within ±15%.\n"
            "\n"
            "READINESS RULE (for your own reasoning):\n"
            "- Consider a story READY if ALL scores ≥ 4 AND there are NO safety concerns.\n"
            "- If the story is NOT ready: include at least one issue and provide clear, "
            "specific edit_instructions for ONE revision pass (≤120 words).\n"
            "- If the story IS ready: issues may be empty and edit_instructions may be very brief.\n"
            "\n"
            "You MUST respond in a form compatible with the StoryFeedback schema provided by the tool."
        )

        self._llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Ask the model to directly emit a StoryFeedback instance
        self._structured_llm = self._llm.with_structured_output(StoryFeedback)

    def review(
        self,
        user_request: str,
        story: str,
        *,
        child_age: int = 7,
        tone: str = "soothing",
        length_target: int = 500,
    ) -> StoryFeedback:
        """
        Evaluate a generated story and return a structured StoryFeedback.

        Readiness check (caller logic, not enforced in schema):
        - Story is considered ready if all scores >= 4 and there are no serious safety issues.
        """
        word_count = len(story.split())

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=(
                    "Review the following bedtime story REQUEST and DRAFTED STORY.\n"
                    "Assess how well the story fulfills the request and is suitable for ages 5-10.\n"
                    "Apply your rubric, readiness rule, and the output schema you were given.\n\n"
                    f"User request: {user_request}\n"
                    f"Child age: {child_age}\n"
                    f"Tone preset: {tone}\n"
                    f"Target length (words): {length_target}\n"
                    f"Draft story word count (approx): {word_count}\n\n"
                    f"--- DRAFT STORY START ---\n{story}\n--- DRAFT STORY END ---"
                )
            ),
        ]

        try:
            feedback: StoryFeedback = self._structured_llm.invoke(messages)
            feedback.metadata = {
                "age": child_age,
                "tone": tone,
                "target_length": length_target,
                "word_count": word_count,
            }
            return feedback

        except Exception as e:
            return self._fallback_feedback(
                error=str(e),
                child_age=child_age,
                tone=tone,
                length_target=length_target,
                word_count=word_count,
            )

    @staticmethod
    def is_ready(feedback: StoryFeedback, threshold: int = 4) -> bool:
        """Compute 'ready' purely from scores (no separate passed flag)."""
        s = feedback.scores
        return min(
            s.age_fit,
            s.safety_sensitivity,
            s.clarity_structure,
            s.tone_bedtime,
            s.engagement_creativity,
            s.length_fit,
        ) >= threshold

    @staticmethod
    def _fallback_feedback(
        *,
        error: str,
        child_age: int,
        tone: str,
        length_target: int,
        word_count: int,
    ) -> StoryFeedback:
        """
        Safe failure object if structured output parsing/validation fails.
        Forces a revision with generic but safe instructions.
        """
        scores = Scores(
            age_fit=3,
            safety_sensitivity=3,
            clarity_structure=3,
            tone_bedtime=3,
            engagement_creativity=3,
            length_fit=3,
        )
        return StoryFeedback(
            scores=scores,
            issues=["judge_json_invalid", error[:200]],
            edit_instructions=(
                "Revise once: simplify vocabulary, shorten sentences (aim ~5-15 words), "
                "ensure only a gentle problem with a kind resolution, remove any scary or intense "
                "elements, and end with a calm, cozy paragraph that decelerates the energy. "
                "Keep total length near the target within ±15%."
            ),
            metadata={
                "age": child_age,
                "tone": tone,
                "target_length": length_target,
                "word_count": word_count,
            },
        )
