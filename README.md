# PDF RAG System

This project implements a PDF Retrieval-Augmented Generation (RAG) system using Jina for embedding and Pinecone for vector storage. The system parses PDF documents, embeds the extracted content, and stores the embeddings in Pinecone for efficient retrieval.

## Project Structure

```
pdf-rag-system
├── src
│   ├── embeddings
│   │   ├── __init__.py
│   │   └── jina_embedder.py
│   ├── parser
│   │   ├── __init__.py
│   │   └── pdf_parser.py
│   ├── storage
│   │   ├── __init__.py
│   │   └── pinecone_store.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── config.py
│   └── main.py
├── data
│   └── pdfs
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd pdf-rag-system
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in the required values, such as API keys for Pinecone and any other necessary configurations.

## Usage

1. Place your PDF files in the `data/pdfs` directory.
2. Run the main application:
   ```bash
   python src/main.py
   ```

## Features

- **PDF Parsing:** Extracts text and metadata from PDF documents.
- **Embedding:** Uses Jina v3 to create embeddings from the parsed content.
- **Storage:** Stores the embeddings in Pinecone for efficient retrieval.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
