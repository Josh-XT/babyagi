import openai
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_embedding(text, chunk_size=4096):
    openai.api_key = OPENAI_API_KEY
    text = text.replace("\n", " ")
    def process_openai_embedding_chunk(chunk_text):
        return openai.Embedding.create(input=[chunk_text], model="text-embedding-ada-002")["data"][0]["embedding"]
    # Split the text into smaller chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        # Get embeddings for each chunk using OpenAI API
    all_embeddings = [process_openai_embedding_chunk(chunk) for chunk in chunks]
    # Average the embeddings
    averaged_embedding = np.mean(all_embeddings, axis=0)
    return averaged_embedding