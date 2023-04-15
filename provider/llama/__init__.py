import subprocess
import torch
import numpy as np
import os
from dotenv import load_dotenv
from transformers import LongformerTokenizer, LongformerModel
load_dotenv()
AI_PROVIDER = os.getenv("AI_PROVIDER")

if AI_PROVIDER == "llama":
    LLAMA_PATH = os.getenv("LLAMA_PATH")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    def custom_embedding(text):
        return get_embedding(text)

def instruct(model, prompt, temperature, max_tokens):
    cmd = [f"{LLAMA_PATH}/main", "-p", prompt]
    result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

def get_embedding(text, chunk_size=4096):
    text = text.replace("\n", " ")
    # Split the text into smaller chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    all_embeddings = []
    for chunk in chunks:
        input_ids = tokenizer.encode(chunk, max_length=chunk_size, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(embeddings[0])
    # Average the embeddings
    averaged_embedding = np.mean(all_embeddings, axis=0)
    return averaged_embedding