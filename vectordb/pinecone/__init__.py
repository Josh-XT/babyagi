import pinecone
import os
from dotenv import load_dotenv
load_dotenv()
VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER", "pinecone")

if VECTORDB_PROVIDER.lower() == "pinecone":
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
    OBJECTIVE = os.getenv("OBJECTIVE")
    # Table config
    table_name = os.getenv("TABLE_NAME", "")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # Create Pinecone index
    dimension = 1536
    metric = "cosine"
    pod_type = "p1"
    if table_name not in pinecone.list_indexes():
        pinecone.create_index(
            table_name, dimension=dimension, metric=metric, pod_type=pod_type
        )
    # Connect to the index
    index = pinecone.Index(table_name)

def results(query, top_results_num):
    results = index.query(query, top_k=top_results_num, include_metadata=True, namespace=OBJECTIVE)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]

def enrich_results(result_id, vector, result, task):
    index.upsert(
        [(result_id, vector, {"task": task["task_name"], "result": result})],
        namespace=OBJECTIVE
    )