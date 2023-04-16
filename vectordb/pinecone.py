import pinecone
from Config import Config
CFG = Config()

class VectorDB:
    def __init__(self):
        if CFG.VECTORDB_PROVIDER.lower() == "pinecone":
            pinecone.init(api_key=CFG.PINECONE_API_KEY, environment=CFG.PINECONE_ENVIRONMENT)
            dimension = 1536
            metric = "cosine"
            pod_type = "p1"
            
            if CFG.TABLE_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    CFG.TABLE_NAME, dimension=dimension, metric=metric, pod_type=pod_type
                )
            self.index = pinecone.Index(CFG.TABLE_NAME)

    def results(self, query, top_results_num):
        results = self.index.query(query, top_k=top_results_num, include_metadata=True, namespace=CFG.OBJECTIVE)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]

    def store_results(self, result_id, vector, result, task):
        self.index.upsert(
            [(result_id, vector, {"task": task["task_name"], "result": result})],
            namespace=CFG.OBJECTIVE
        )
