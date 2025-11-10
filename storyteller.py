import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class StoryTeller:
    def __init__(self, system_prompt_path: str = "system_prompt.txt", max_tokens: int = 3000, temperature: float = 0.1):
        """
        Initialize the StoryTeller with a system prompt from a file.
        
        Args:
            system_prompt_path: Path to the system prompt text file
            max_tokens: Maximum tokens for the LLM response
            temperature: Temperature setting for the LLM
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
        self.message_history = [SystemMessage(content=self.system_prompt)]
    
    def _load_system_prompt(self, file_path: str) -> str:
        """Load the system prompt from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found: {file_path}")
    
    def tell_story(self, user_request: str) -> str:
        """
        Generate a story based on the user's request.
        
        Args:
            user_request: The user's story request
            
        Returns:
            The generated story
        """
        user_message = HumanMessage(content=user_request)
        self.message_history.append(user_message)
        response = self.llm.invoke(self.message_history)
        self.message_history.append(response)
        return response.content

