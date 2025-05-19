# query.py
import argparse
import logging
from ollama import chat
from ollama import ChatResponse
from src.embeddings.jina_embedder import JinaEmbedder
from src.storage.pinecone_store import PineconeStore
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rag_system():
    """Initialize and set up the RAG system components"""
    if not Config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY environment variable not set")
        return None, None
    
    # Initialize embedder
    embedder = JinaEmbedder()
    
    # Using a small default text to extract dimension
    sample_embedding = embedder.embed_documents(["This is a sample text"])
    embedder.dimension = len(sample_embedding[0])
    
    # Initialize Pinecone store
    vector_store = (
        PineconeStore(api_key=Config.PINECONE_API_KEY)
        .connect_embedder(embedder)
    )
    
    # Get the index (assuming it already exists)
    vector_store.index = vector_store.pc.Index(vector_store.index_name)
    
    return embedder, vector_store

def query_documents(query_text, top_k=5):
    """Query the vector store for relevant documents"""
    embedder, vector_store = setup_rag_system()
    if not vector_store:
        return
    
    print(f"Querying with: {query_text}")
    
    # Embed the query
    query_embedding = embedder.embed_documents([query_text])[0]
    
    # Query Pinecone
    results = vector_store.index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    # Extract and format relevant passages
    passages = []
    for i, match in enumerate(results["matches"]):
        # Get source information
        source = match['metadata'].get('source', 'Unknown')
        chunk_index = match['metadata'].get('chunk_index', 'Unknown')
        score = match['score']
        
        # Get text content (if available)
        text = match['metadata'].get('text', '')
        if not text:
            continue  # Skip if no text is available
            
        # Format the passage with source information
        passage = {
            "text": text,
            "source": source,
            "chunk_index": chunk_index,
            "score": score
        }
        passages.append(passage)

    # Format context for the model
    context = "\n\n".join([
        f"[Source: {p['source']}, Score: {p['score']:.4f}]\n{p['text']}" 
        for p in passages
    ])

    # Prepare prompt for Ollama
    prompt = f"""You are a helpful AI assistant answering questions based on the provided context.
Answer the question using ONLY the information from the context. If the context doesn't contain 
the information needed to answer the question, say "I don't have enough information to answer this question."

CONTEXT:
{context}

QUESTION: {query_text}

ANSWER:"""
        
    
    # Call Ollama model with passage as context
    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])  
    print(response)

    print(f"Answer: {response.message.content}")
        
    return response.message.content

def interactive_query():
    """Interactive query mode"""
    print("\nWelcome to the RAG Query System!")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
        
        query_documents(query)

def main():
    parser = argparse.ArgumentParser(description="Query RAG System")
    parser.add_argument('--query', type=str, help='Single query to run')
    parser.add_argument('--interactive', action='store_true', help='Start interactive query mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    if args.query:
        query_documents(args.query, args.top_k)
    elif args.interactive:
        interactive_query()
    else:
        # Default to interactive mode
        interactive_query()

if __name__ == "__main__":
    main()