import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "data/pdfs")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embedding-model")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")