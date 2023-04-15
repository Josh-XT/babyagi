import requests
import torch
import numpy as np
import os
from dotenv import load_dotenv
from transformers import LongformerTokenizer, LongformerModel
load_dotenv()
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")

if AI_PROVIDER == "oobabooga":
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    def custom_embedding(text):
        return get_embedding(text)

def chat(model, prompt, temperature, max_tokens):
    response = requests.post("http://localhost:7860/run/textgen", json={
        "data": [
            [
                prompt,
                {
                    'max_new_tokens': max_tokens,
                    'do_sample': True,
                    'temperature': temperature,
                    'top_p': 0.73,
                    'typical_p': 1,
                    'repetition_penalty': 1.1,
                    'encoder_repetition_penalty': 1.0,
                    'top_k': 0,
                    'min_length': 0,
                    'no_repeat_ngram_size': 0,
                    'num_beams': 1,
                    'penalty_alpha': 0,
                    'length_penalty': 1,
                    'early_stopping': False,
                    'seed': -1,
                    'add_bos_token': True,
                    'custom_stopping_strings': [],
                    'truncation_length': 2048,
                    'ban_eos_token': False,
                }
            ]
        ]
    }).json()
    data = response['data'][0]
    return data.replace("\\n", "\n").strip()

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