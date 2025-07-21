# Couple Counselor AI WebSocket

A professional couple counselor AI assistant built with FastAPI, WebSocket, and OpenAI GPT models. Features a RAG (Retrieval-Augmented Generation) system using ChromaDB with OpenAI embeddings for enhanced knowledge-based responses.

## Features

- Real-time chat via WebSocket
- OpenAI GPT-4 powered responses
- RAG system with OpenAI embeddings
- PDF and TXT file ingestion for knowledge base
- MongoDB for chat history storage
- ChromaDB for vector storage and semantic search

## Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=your_mongodb_connection_string_here
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The server will start on `http://127.0.0.1:3000`

