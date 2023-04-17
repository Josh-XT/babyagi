# SuperPrompter

The `SuperPrompter` class is a Python class that helps you generate text responses to a given task using AI language models. It is built on top of the OpenAI API and Hugging Face's Transformers library.

The `SuperPrompter` class is designed to be flexible and easy to use. It can take as input a single task or a list of tasks, and it can process text files or URLs. The output of the class is a generated response to the task.
## Installation

To use the `SuperPrompter` class, you'll need to install the following dependencies:
- chromadb
- selenium
- webdriver-manager

You can install these dependencies using pip:

```bash

pip install chromadb selenium webdriver-manager
```


## Usage

Here's an example of how to use the `SuperPrompter` class:

```python

from SuperPrompter import SuperPrompter

task = "Perform the task with the given context."
folder_path = "/path/to/your/folder"
url = "https://example.com"

sp = SuperPrompter(task, folder_path, url)
print("Response:", sp.response)
```



In this example, we're creating a new `SuperPrompter` object with a single task and a folder path and a URL. The `response` property of the `SuperPrompter` object contains the generated response to the task.
## Contributing

Contributions to the `SuperPrompter` class are welcome! If you'd like to contribute, please open a pull request or an issue on the GitHub repository.
