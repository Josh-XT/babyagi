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
        self.AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
        self.VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER", "pinecone")
        self.EMBEDDING = os.getenv("EMBEDDING", "openai")
        self.AI_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
        self.AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.4))
        
        # Extensions Configuration

        # OpenAI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # Pinecone
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        self.TABLE_NAME = os.getenv("TABLE_NAME")
