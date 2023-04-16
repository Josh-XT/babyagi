import os
import pinecone
from dotenv import load_dotenv

class PineconeVectorDB:
    def __init__(self):
        load_dotenv()
        self.VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER", "pinecone")
        self.OBJECTIVE = os.getenv("OBJECTIVE")

        if self.VECTORDB_PROVIDER.lower() == "pinecone":
            self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
            self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
            self.TABLE_NAME = os.getenv("TABLE_NAME", "")
            
            pinecone.init(api_key=self.PINECONE_API_KEY, environment=self.PINECONE_ENVIRONMENT)
            dimension = 1536
            metric = "cosine"
            pod_type = "p1"
            
            if self.TABLE_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    self.TABLE_NAME, dimension=dimension, metric=metric, pod_type=pod_type
                )
            self.index = pinecone.Index(self.TABLE_NAME)

    def results(self, query, top_results_num):
        results = self.index.query(query, top_k=top_results_num, include_metadata=True, namespace=self.OBJECTIVE)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]

    def store_results(self, result_id, vector, result, task):
        self.index.upsert(
            [(result_id, vector, {"task": task["task_name"], "result": result})],
            namespace=self.OBJECTIVE
        )
