from transformers import AutoModel

class JinaEmbedder:
    def __init__(self):
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

    
    def embed_documents(self, documents):
        """
        Embed a list of documents using Jina v3 embedding model
        
        Args:
            documents (List[str]): List of document texts to embed
            
        Returns:
            List[List[float]]: List of document embeddings
        """
        embeddings = self.model.encode(
            documents, 
            task="text-matching"
        )
        
        # Convert to list of lists if numpy array
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
            
        return embeddings