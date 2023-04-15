import os
from langchain.vectorstores import FAISS
from transformers import LongformerTokenizer, LongformerModel
from dotenv import load_dotenv
load_dotenv()
vectordb = os.getenv("VECTORDB_PROVIDER", "FAISS")
OBJECTIVE = os.getenv("OBJECTIVE")
if vectordb == "FAISS":
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    index = FAISS.from_texts(
        texts=["_"],
        embedding=model,
        metadatas=[{"task": os.getenv("INITIAL_TASK")}]
    )

def results(query, top_results_num):
    results = index.query(query, top_k=top_results_num, include_metadata=True, namespace=OBJECTIVE)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]

def store_results(result_id, vector, result, task):
    index.add_with_ids([vector], [result_id])
    index.metadata.add(result_id, {"task": task["task_name"], "result": result})