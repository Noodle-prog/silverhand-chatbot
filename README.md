# Gradio Examples for Advanced Topics in CG 

This is a repository for an LLM Chatbot (Johnny Silverhand) using LM Studio, Gradio and RAG with a ChromaDB.

## Setup

1. `uv sync`
2. For the RAG example, run `python index-data.py` first. It will download a few files from the Cyberpunk 2077 Fandom Wiki, convert them to markdown and index them in a local vector store.
3. Run specific examples with `uv run <file>` (e.g., `uv run chat.py`). This will open a gradio interface at http://localhost:7860 for you to prompt.

For the chat examples, you'll need to have an LM Studio server running with the model "qwen3.5-4b" open at http://localhost:1234/v1/