import os
from dotenv import load_dotenv
load_dotenv()

class Config():
    def __init__(self):
        # General Configuration
        self.AGENT_NAME = os.getenv("AGENT_NAME", "Agent-LLM")
        
        # Goal Configuation
        self.OBJECTIVE = os.getenv("OBJECTIVE", "Solve world hunger")
        self.INITIAL_TASK = os.getenv("INITIAL_TASK", "Develop a task list")
        
        # AI Configuration
        self.AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()

        # Model configuration
        self.AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo").lower()
        self.AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.4))
        self.MAX_TOKENS = os.getenv("MAX_TOKENS", 2000)
        
        # Extensions Configuration

        # OpenAI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
