import os
from src.parser.pdf_parser import parse_pdf
from src.embeddings.jina_embedder import JinaEmbedder
from src.storage.pinecone_store import PineconeStore
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks of specified size"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            # Last chunk
            chunk = text[start:text_length]
        else:
            # Find a good breakpoint (period, newline, etc.)
            breakpoint = text.rfind(". ", start + chunk_size - 200, end)
            if breakpoint == -1:
                breakpoint = text.rfind("\n", start + chunk_size - 200, end)
            if breakpoint == -1:
                breakpoint = end
            else:
                breakpoint += 2  # Include the period and space
            
            chunk = text[start:breakpoint]
        
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = start + chunk_size - overlap
    
    return chunks

def main():
    pdf_directory = 'data'

    if not Config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY environment variable not set")
        return
    
    embedder = JinaEmbedder()

    # Using a small default text to extract dimension
    sample_embedding = embedder.embed_documents(["This is a sample text"])
    embedder.dimension = len(sample_embedding[0])

    # Initialize Pinecone store
    vector_store = (
        PineconeStore(api_key=Config.PINECONE_API_KEY)
        .connect_embedder(embedder)
        .create_index_if_not_exists()
    )

    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            logger.info(f"Processing file: {pdf_path}")
            
            # Parse the PDF to extract text
            full_text = parse_pdf(pdf_path)

            # Chunk the text
            text_chunks = chunk_text(full_text)
            logger.info(f"Created {len(text_chunks)} chunks from {pdf_file}")

            # Prepare data for vector store
            texts = text_chunks
            ids = [f"{pdf_file}_{i}" for i in range(len(text_chunks))]
            metadata = [{"source": pdf_file, "chunk_index": i} for i in range(len(text_chunks))]
            
            # Add document to vector store
            vector_count = vector_store.add_documents(texts, ids, metadata)
            logger.info(f"Stored {vector_count} embeddings for: {pdf_file}")
        
        logger.info("Processing complete")

if __name__ == "__main__":
    main()