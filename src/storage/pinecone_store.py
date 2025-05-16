from pinecone import Pinecone, ServerlessSpec
import os
import time

class PineconeStore:
    def __init__(self, api_key, index_name="jina-embeddings"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.embedder = None
        self.index = None

    def connect_embedder(self, embedder):
        """
        Connect the embedder to the Pinecone store.
        
        Args:
            embedder: The embedder instance to connect.
        """
        self.embedder = embedder
        return self
    
    def create_index_if_not_exists(self):
        """
        Create the Pinecone index if it does not exist.
        """
        if not self.embedder:
            raise ValueError("Embedder is not connected. Please connect an embedder first.")

        dimension = self.embedder.dimension

        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating new index '{self.index_name}' with dimension {dimension}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
            
            # Wait for index to be ready
            time.sleep(1)
        else:
            print(f"Using existing index '{self.index_name}'")
        
        self.index = self.pc.Index(self.index_name)
        return self
    
    def add_documents(self, texts, ids=None, metadata=None):
        """
        Embed and add documents to Pinecone
        
        Args:
            texts (List[str]): List of document texts
            ids (List[str], optional): List of document IDs
            metadata (List[dict], optional): List of metadata dicts for each document
        """

        if not self.embedder:
            raise ValueError("Must connect an embedder before adding documents")
            
        if not self.index:
            self.create_index_if_not_exists()
            
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
            
        # Generate empty metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]
            
        # Ensure lists are of same length
        assert len(texts) == len(ids) == len(metadata), "texts, ids, and metadata must be of same length"
        
        # Embed the documents
        embeddings = self.embedder.embed_documents(texts)

        # Prepare vectors for upsert
        vectors = [
            {"id": ids[i], "values": embeddings[i], "metadata": {"text": texts[i], **metadata[i]}}
            for i in range(len(texts))
        ]

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
            
        return len(vectors)