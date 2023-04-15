#!/usr/bin/env python3
import os
import subprocess
import time
import requests
import numpy as np
from collections import deque
from typing import Dict, List
import importlib
import torch
from transformers import LongformerTokenizer, LongformerModel
from langchain.vectorstores import FAISS
import openai
import pinecone
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# Set your AI Model to use. You can use any of the models listed here:
# Set to "ooba" for the Oobabooga Text Generation Web UI server
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")

# Engine configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# If we have an API key, use the OpenAI API
if OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"
    openai.api_key = OPENAI_API_KEY
    if "gpt-4" in AI_MODEL.lower():
        print(
            "\033[91m\033[1m"
            + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
            + "\033[0m\033[0m"
        )
else:
    # Otherwise, use the local engine
    print("\033[91m\033[1m" + "\n*****USING LOCAL ENGINE*****" + "\033[0m\033[0m")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.0))

# Extensions support begin

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, AI_MODEL, DOTENV_EXTENSIONS = parse_arguments()

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions # but also provide command line
# arguments to override them

# Extensions support end

# Check if we know what we are doing
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"
assert INITIAL_TASK, "INITIAL_TASK environment variable is missing from .env"

# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")

# Configure Vector DB
if PINECONE_API_KEY:
    # Table config
    YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
    assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # Create Pinecone index
    table_name = YOUR_TABLE_NAME
    dimension = 1536
    metric = "cosine"
    pod_type = "p1"
    if table_name not in pinecone.list_indexes():
        pinecone.create_index(
            table_name, dimension=dimension, metric=metric, pod_type=pod_type
        )
    # Connect to the index
    index = pinecone.Index(table_name)
else:
    index = FAISS.from_texts(
    texts=["_"],
    embedding=model,
    metadatas=[{"task": INITIAL_TASK}]
)

# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)

def get_embedding(text, chunk_size=4096):
    text = text.replace("\n", " ")
    def process_openai_embedding_chunk(chunk_text):
        return openai.Embedding.create(input=[chunk_text], model="text-embedding-ada-002")["data"][0]["embedding"]
    
    # Split the text into smaller chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    if OPENAI_API_KEY:
        # Get embeddings for each chunk using OpenAI API
        all_embeddings = [process_openai_embedding_chunk(chunk) for chunk in chunks]
    else:
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

def ai_call(
    prompt: str,
    model: str = AI_MODEL,
    temperature: float = AI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return result.stdout.strip()
            if model == "ooba":
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
            elif not model.startswith("gpt-"):
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
        except:
            print("The API rate limit has been exceeded. Waiting 10 seconds and trying again.")
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    if AI_MODEL == "ooba":
            prompt = f"""
You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
This is not a conversation, perform the task and return the result.
The last completed task has the result: {result}.
This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
Return the tasks as an array.
This is not a conversation, perform the task and return the results.
Results:"""
    else:
        prompt = f"""
You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
The last completed task has the result: {result}.
This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
Return the tasks as an array."""
    response = ai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]

def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    if AI_MODEL == "ooba":
        prompt = f"""
You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
Consider the ultimate objective of your team:{OBJECTIVE}.
Do not remove any tasks. Return the result as a numbered list, like:
#. First task
#. Second task
Start the task list with number {next_task_id}.
This is not a conversation, perform the task and return the results.
Results:"""
    else:
        prompt = f"""
You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
Consider the ultimate objective of your team:{OBJECTIVE}.
Do not remove any tasks. Return the result as a numbered list, like:
#. First task
#. Second task
Start the task list with number {next_task_id}."""
    response = ai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

def execution_agent(objective: str, task: str) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """
    
    context = context_agent(query=objective, top_results_num=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    if AI_MODEL == "ooba":
        prompt = f"""
You are an AI who confidently performs one task based on the following objective: {objective}.
Take into account these previously completed tasks: {context}.
Your task to perform confidently: {task}.
This is not a conversation, perform the task and return the results.
Results:"""
    else:
        prompt = f"""
You are an AI who performs one task based on the following objective: {objective}\n.
Take into account these previously completed tasks: {context}\n.
Your task: {task}\nResponse:"""
    return ai_call(prompt, max_tokens=2000)

def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    query_embedding = get_embedding(query)
    if PINECONE_API_KEY:
        results = index.query(query_embedding, top_k=top_results_num, include_metadata=True, namespace=OBJECTIVE)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]
    else:
        # Use FAISS
        results = index.similarity_search_with_score(query, k=top_results_num)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return [item[0].metadata["task"] for item in sorted_results]    

# Add the first task
first_task = {"task_id": 1, "task_name": INITIAL_TASK}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in vector db
        enriched_result = {
            "data": result
        }  # This is where you should enrich the result if needed
        if PINECONE_API_KEY:
            result_id = f"result_{task['task_id']}"
            vector = get_embedding(enriched_result["data"])  # get vector of the actual result extracted from the dictionary
            index.upsert(
                [(result_id, vector, {"task": task["task_name"], "result": result})],
            namespace=OBJECTIVE
            )
        else:
            # Use FAISS
            result_id = f"result_{task['task_id']}"
            vector = get_embedding(enriched_result["data"])
            index.add_with_ids([vector], [result_id])
            index.metadata.add(result_id, {"task": task["task_name"], "result": result})

        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list],
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again
