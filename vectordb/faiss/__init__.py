import os
from langchain.vectorstores import FAISS
from transformers import LongformerTokenizer, LongformerModel
from dotenv import load_dotenv

class FaissVectorDB:
    def __init__(self):
        load_dotenv()
        self.OBJECTIVE = os.getenv("OBJECTIVE")
        self.VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER", "faiss")

        if self.VECTORDB_PROVIDER.lower() == "pinecone":
            tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
            model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            self.index = FAISS.from_texts(
                texts=["_"],
                embedding=model,
                metadatas=[{"task": os.getenv("INITIAL_TASK")}]
            )

    def results(self, query, top_results_num):
        results = self.index.query(query, top_k=top_results_num, include_metadata=True, namespace=self.OBJECTIVE)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]

    def store_results(self, result_id, vector, result, task):
        self.index.add_with_ids([vector], [result_id])
        self.index.metadata.add(result_id, {"task": task["task_name"], "result": result})
