# 🤖 Multi-Agent Coding Assistant

A powerful, entirely autonomous Python coding assistant built with **Streamlit**. It uses a multi-agent architecture where a central Orchestrator delegates tasks to specialized subagents.

## ✨ Features
* **Orchestrator Agent**: Breaks down the user's coding request and manages the workflow.
* **Planner Agent**: Generates a step-by-step technical architecture and implementation plan.
* **Coder Agent**: Writes clean, modular code based on the Planner's exact specifications.
* **Reviewer Agent**: Reviews the code for bugs, missing type hints, and bad practices.
* **Autonomous Tester (Sandbox)**: Extracts the generated Python code, runs it in an isolated local subprocess, captures any syntax errors or stack traces, and automatically feeds the errors back to the Reviewer for autonomous self-correction!

## 🚀 Quickstart

1. **Clone the repository**
```bash
git clone https://github.com/DS123-ally/Multi-Coding-Assistant.git
cd Multi-Coding-Assistant
```

2. **Install dependencies**
It is highly recommended to use `uv` for lightning-fast installation:
```bash
uv venv --python 3.12
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```
*(Alternatively, use standard `python -m venv venv` and `pip install -r requirements.txt`)*

3. **Configure your LLM**
Copy the `.env` file or configure the variables directly in the Streamlit Sidebar.
By default, the app is configured to look for a **local Ollama** instance running `gemma4:e4b`.
- **API Base URL**: `http://localhost:11434/v1`
- **Model Name**: `gemma4:e4b` (or your preferred local tag)

*To use a cloud provider like Groq or OpenAI, simply change the Base URL and input your API key!*

4. **Run the App**
```bash
streamlit run app.py
```

## 🦙 About the Gemma Model

This project is optimized to run locally using **Gemma**, an open-weights model by Google. Specifically, it is configured by default for the `gemma4:e4b` variant via **Ollama**.

- **Why Gemma?**: Gemma models are lightweight and highly capable, making them perfect for running on local hardware without sacrificing coding intelligence.
- **Privacy First**: By running Gemma locally through Ollama, none of your prompts or proprietary code are sent to external cloud servers. The entire multi-agent workflow happens directly on your machine.
- **Handling Local Quirks**: Local LLMs can sometimes output unexpected markdown or syntax. To handle this, the system prompts in this app are strictly engineered to enforce valid Python syntax, and the **Autonomous Tester** acts as a safety net to catch and fix hallucinations.

## 🧠 Architecture
The application is purely Python-based.
- `app.py`: Contains the Streamlit frontend UI and session state management.
- `agents.py`: Contains the backend LLM wrapper, prompt engineering for each agent, and the deterministic `subprocess` sandbox for the Tester feedback loop.
