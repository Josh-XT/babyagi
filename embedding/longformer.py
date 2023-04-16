import torch
import numpy as np
from transformers import LongformerTokenizer, LongformerModel

class Embedding:
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        # Split the text into smaller chunks
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        all_embeddings = []
        for chunk in chunks:
            input_ids = self.tokenizer.encode(chunk, max_length=self.chunk_size, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(input_ids)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                all_embeddings.append(embeddings[0])
        # Average the embeddings
        averaged_embedding = np.mean(all_embeddings, axis=0)
        return averaged_embedding
