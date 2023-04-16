import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

def instruct(model, prompt, temperature, max_tokens):
    cmd = [f"llama/main", "-p", prompt]
    result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()
