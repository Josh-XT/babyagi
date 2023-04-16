import openai
import numpy as np
from Config import Config
CFG = Config()
class Embedding:
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size
        openai.api_key = CFG.OPENAI_API_KEY

    def get_embedding(self, text):
        text = text.replace("\n", " ")

        def process_openai_embedding_chunk(chunk_text):
            return openai.Embedding.create(input=[chunk_text], model="text-embedding-ada-002")["data"][0]["embedding"]

        # Split the text into smaller chunks
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # Get embeddings for each chunk using OpenAI API
        all_embeddings = [process_openai_embedding_chunk(chunk) for chunk in chunks]

        # Average the embeddings
        averaged_embedding = np.mean(all_embeddings, axis=0)

        return averaged_embedding
