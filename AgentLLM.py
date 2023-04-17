import os
import chromadb
import importlib
import secrets
import string
from chromadb.utils import embedding_functions
from Config import Config
from typing import List
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from Commands import Commands
class AgentLLM:
    def __init__(self):
        self.CFG = Config()
        if self.CFG.AI_PROVIDER == "openai":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=self.CFG.OPENAI_API_KEY)
        else:
            self.embedding_function = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl")
        self.chroma_persist_dir = "memories"
        self.chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.chroma_persist_dir,
            )
        )
        # The collection of thoughts is associated with the AGENT_NAME
        # May change to a unique session variable later.
        self.collection = self.chroma_client.get_or_create_collection(
            name=str(self.CFG.AGENT_NAME).lower(),
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function,
        )
        ai_module = importlib.import_module(f"provider.{self.CFG.AI_PROVIDER}")
        self.ai_instance = ai_module.AIProvider()
        self.instruct = self.ai_instance.instruct

    def run(self, task: str, folder_path: str = None, url: str = None):
        responses = self.process_input(task, folder_path=folder_path, url=url)
        for response in responses:
            self.store_result(task, response)
        context = self.context_agent(query=task, top_results_num=5)
        prompt = self.get_prompt_with_context(task=task, context=context)
        self.response = self.instruct(prompt)
        self.store_result(task, self.response)
        print(f"Response: {self.response}")
        return self.response

    def store_result(self, task_name: str, result: str):
        result_id = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(64))
        if (len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0):
            self.collection.update(ids=result_id, documents=result, metadatas={"task": task_name, "result": result})
        else:
            self.collection.add(ids=result_id, documents=result, metadatas={"task": task_name, "result": result})

    def context_agent(self, query: str, top_results_num: int) -> List[str]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(query_texts=query, n_results=min(top_results_num, count), include=["metadatas"])
        return [item["result"] for item in results["metadatas"][0]]

    def get_prompt_with_context(self, task: str, context: List[str]) -> str:
        context_str = "\n\n".join(context)
        prompt = f"Context: {context_str}\n\nTask: {task}\n\nResponse:"
        return prompt

    def chunk_content(self, content: str, max_length: int = 500) -> List[str]:
        content_chunks = []
        content_length = len(content)
        for i in range(0, content_length, max_length):
            chunk = content[i:i + max_length]
            content_chunks.append(chunk)
        return content_chunks

    def process_chunks(self, task: str, chunks: List[str]) -> List[str]:
        responses = []
        for chunk in chunks:
            prompt = self.get_prompt_with_context(task=task, context=[chunk])
            response = self.instruct(prompt)
            responses.append(response)
        return responses
    
    # Data reading extensions
    def scrape_website(self, url: str) -> str:
        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.get(url)
        content = driver.page_source
        driver.quit()
        return content
    
    def process_folder(self, folder_path: str, task: str) -> List[str]:
        all_responses = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    content = file.read()
                chunks = self.chunk_content(content)
                responses = self.process_chunks(task, chunks)
                all_responses.extend(responses)
        return all_responses
    
    def process_input(self, task: str, folder_path: str = None, url: str = None) -> List[str]:
        all_responses = []
        if folder_path:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as file:
                        content = file.read()
                    chunks = self.chunk_content(content)
                    responses = self.process_chunks(task, chunks)
                    all_responses.extend(responses)
        if url:
            content = self.scrape_website(url)
            chunks = self.chunk_content(content)
            responses = self.process_chunks(task, chunks)
            all_responses.extend(responses)
        return all_responses
    
if __name__ == "__main__":
    task = "Perform the task with the given context."
    # Folder path to read files from
    folder_path = "/path/to/your/folder"
    # URL to scrape
    url = "https://example.com"

    sp = AgentLLM(task, folder_path, url)
    print("Response: ", sp.response)