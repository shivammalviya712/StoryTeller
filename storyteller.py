import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()


class StoryTeller:
    def __init__(
        self,
        system_prompt_path: str = "system_prompt.txt",
        max_tokens: int = 3000,
        temperature: float = 0.1,
    ):
        """
        Initialize the StoryTeller with a system prompt from a file.

        Args:
            system_prompt_path: Path to the system prompt text file.
            max_tokens: Maximum tokens for the LLM response.
            temperature: Temperature setting for the LLM.
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Keep message history between calls, as requested.
        self.history = InMemoryChatMessageHistory()

    def _load_system_prompt(self, file_path: str) -> str:
        """Load the system prompt from a text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found: {file_path}")

    def tell_story(self, user_request: str) -> str:
        """
        Generate a story based on the user's request.

        Args:
            user_request: The user's story request.

        Returns:
            The generated story.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            *self.history.messages,
            HumanMessage(content=user_request),
        ]
        response = self.llm.invoke(messages)
        # Persist this interaction in history for future turns.
        self.history.add_user_message(user_request)
        self.history.add_ai_message(response.content)
        return response.content

    def revise_story(self, user_request: str, draft_story: str, feedback: str) -> str:
        """
        Revise an existing story using feedback from a judge.

        Args:
            user_request: The original user request for context.
            draft_story: The story that was previously generated.
            feedback: Feedback describing what should be adjusted (typically edit_instructions).

        Returns:
            The revised story content.
        """
        revision_prompt = (
            "You previously drafted a bedtime story. Refine it based on the provided "
            "feedback while keeping the original request in mind. The story must stay "
            "age-appropriate, comforting, and engaging for children ages 5-10. "
            "Keep the overall plot, characters, and approximate length similar unless "
            "the feedback explicitly asks for a bigger change.\n\n"
            f"Original request:\n{user_request}\n\n"
            f"Draft story:\n{draft_story}\n\n"
            f"Feedback to apply:\n{feedback}\n\n"
            "Provide the improved story only."
        )
        messages = [
            SystemMessage(content=self.system_prompt),
            *self.history.messages,
            HumanMessage(content=revision_prompt),
        ]
        response = self.llm.invoke(messages)
        # Persist this revision turn in history as well.
        self.history.add_user_message(revision_prompt)
        self.history.add_ai_message(response.content)
        return response.content
