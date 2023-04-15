import openai
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
AI_PROVIDER = os.getenv("AI_PROVIDER")
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if AI_PROVIDER == "openai":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"
    openai.api_key = OPENAI_API_KEY
    if "gpt-4" in AI_MODEL.lower():
        print(
            "\033[91m\033[1m"
            + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
            + "\033[0m\033[0m"
        )

def instruct(model, prompt, temperature, max_tokens):
    if not model.startswith("gpt-"):
        # Use completion API
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text.strip()
    else:
        # Use chat completion API
        messages = [{"role": "system", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()

def get_embedding(text, chunk_size=4096):
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