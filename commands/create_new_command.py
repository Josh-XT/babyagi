from typing import List
from Commands import Commands
from Config import Config
from AgentLLM import AgentLLM

CFG = Config()

class create_new_command(Commands):
    def __init__(self):
        self.commands = {
            "Create a new command": self.create_command
        }

    def create_command(self, function_description: str) -> List[str]:
        args = [function_description]
        function_string = """
from typing import List
from Commands import Commands
from AgentLLM import AgentLLM

class code_evaluation(Commands):
    def __init__(self):
        self.commands = {
            "Evaluate Code": self.evaluate_code
        }

    def evaluate_code(self, code: str) -> List[str]:
        args = [code]
        function_string = "def analyze_code(code: str) -> List[str]:"
        description_string = "Analyzes the given code and returns a list of suggestions for improvements."
        prompt = f"You are now the following python function: ```# {description_string}\n{function_string}```\n\nOnly respond with your `return` value. Args: {args}"
        return AgentLLM().run(prompt, commands_enabled=False)
        """
        description_string = """You write new commands for this framework. Ensure commands summaries are short and concice in self.commands.
1. File name.
2. Code for the new command that fits in this framework."""
        prompt = f"{description_string}\n{function_string}```\n\nOnly respond with your `return` values. Args: {args}"
        response = AgentLLM().run(prompt, commands_enabled=False)
        # Parse the response, 1 = file name, 2 = code
        file_name = response.split("1. ")[1].split("\n")[0]
        code = response.split("2. ")[1]
        code = code.replace("```", "")
        # Create a new file in the commands directory
        with open(f"commands/{file_name}.py", "w") as f:
            f.write(code)
        return f"Created new command: {file_name}"