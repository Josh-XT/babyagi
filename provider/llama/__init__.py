import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

LLAMA_PATH = os.getenv("LLAMA_PATH")

def instruct(model, prompt, temperature, max_tokens):
    cmd = [f"{LLAMA_PATH}/main", "-p", prompt]
    result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()
