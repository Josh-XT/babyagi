import openai
import numpy as np

class OpenAIEmbedding:
    def __init__(self, api_key, chunk_size=4096):
        self.chunk_size = chunk_size
        self.api_key = api_key
        openai.api_key = self.api_key

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
