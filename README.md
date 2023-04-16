# Agent-LLM
## AI-Powered Task Management System

This repository contains a Python script for an AI-powered task management system that uses Large Language Models (LLMs) to create, prioritize, and execute tasks. The system is designed to create new tasks based on the results of previous tasks and a predefined objective. It uses the natural language processing (NLP) capabilities of various LLMs and your choice of Vector Database to store and retrieve task results for context. This version is based on the original [BabyAGI](https://github.com/yoheinakajima/babyagi)  but adds support for different VectorDB providers, LLM providers, models, and custom prompting per model.
## Features
- AI-powered task creation, prioritization, and execution
- Support for different LLM providers and models
- Custom prompting per model
- Context-based task execution using VectorDB
- Flexible configuration with environment variables, command line arguments, and extension support
- Plugin support for custom AI modules, VectorDB modules, and embedding modules
## Installation
1. Clone this repository:

```bash

git clone https://github.com/Josh-XT/Agent-LLM
cd Agent-LLM
```


1. Install the required Python libraries:

```

pip install -r requirements.txt
```

 
1. Set up your `.env` file with the required environment variables. You can use the provided `example.env` as a starting point.
2. Run the main script:

```css

python main.py
```


## Configuration

The script can be configured through environment variables in the `.env` file, command line arguments, or dotenv extensions. You can set your preferred AI_PROVIDER, VECTORDB_PROVIDER, AI_MODEL, OBJECTIVE, and INITIAL_TASK, among other settings.

For more advanced configuration, you can enable command line arguments or dotenv extensions, which can override any environment variables.
### Adding Custom Plugins

Follow the instructions in the [Adding Custom Plugins](/blob/main/PLUGINS.md)  section to add custom AI modules, VectorDB modules, and embedding modules to Agent-LLM.
### Provider-specific Instructions

Instructions for setting up and configuring each AI provider and VectorDB provider can be found in their respective folders within the "provider" and "vectordb" directories. For example, to set up the "openai" AI provider, refer to the README file within the "provider/openai" folder. Similarly, for setting up the "Pinecone" VectorDB provider, consult the README file within the "vectordb/Pinecone" folder.
### Environment Variables 
- `AI_PROVIDER`: The AI provider to use, e.g., "openai". (Default: "openai") 
- `VECTORDB_PROVIDER`: The Vector Database provider to use, e.g., "Pinecone". (Default: "Pinecone") 
- `EMBEDDING_PROVIDER`: The Embedding provider to use, e.g., "longformer". (Default: "longformer") 
- `AI_MODEL`: The AI model to use, e.g., "gpt-3.5-turbo". (Default: "gpt-3.5-turbo") 
- `OBJECTIVE`: The objective or goal for the AI to achieve. 
- `INITIAL_TASK`: The first task to be executed by the AI. 
- `AI_TEMPERATURE`: The AI's temperature setting, affecting randomness in responses. (Default: 0.0) 
- `ENABLE_COMMAND_LINE_ARGS`: Enable or disable the use of command line arguments. (Default: "false")
## Adding Custom Plugins

To add custom plugins for AI modules, VectorDB modules, or embedding modules, follow these steps: 
- 1. **Create a new folder for your custom plugin** : Create a new folder inside the corresponding directory (`provider` for AI modules, `vectordb` for VectorDB modules, or `embedding` for embedding modules). Name the folder according to your custom plugin. 
1. **Implement the required classes and methods** : In the new folder, create an `__init__.py` file and any other necessary Python files. Implement the required classes and methods based on the specific module you are adding. For example, for AI modules, you should implement a class that extends the base `AIProvider` class and overrides its methods. 
2. **Add a README file** : Add a README.md file to your custom plugin's folder, detailing how to set up and use your custom plugin with Agent-LLM. Include any necessary configuration steps, such as setting environment variables or installing additional dependencies. 
3. **Update Agent-LLM's configuration** : In the `.env` file or through command line arguments, set the corresponding environment variables (`AI_PROVIDER`, `VECTORDB_PROVIDER`, or `EMBEDDING_PROVIDER`) to the name of your custom plugin's folder. 
4. **Test your custom plugin** : Run the main script (`python main.py`) and verify that your custom plugin is working as expected. Make any necessary adjustments to your plugin's code and configuration to ensure proper functionality.
## Contributing

Contributions to this project are welcome! Please feel free to open an issue or submit a pull request if you have any improvements, bug fixes, or feature requests.
