import logging
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import sys
import os

# Add project root to path to import from other modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from chunking_service.document_processor import process_documents, process_documents_v2, process_documents_pdf
from embedding_service.embedding_processor import embedding

# Global cache for FAISS vector store
_cached_vector_store: Optional["FAISSVectorStore"] = None

class State(TypedDict):
    conversation_history: list
    input: str
    messages: list
    output: str
    task_completed: bool
    retrieved_answers: int

class FAISSVectorStore:
    """
    FAISS Vector Store for RAG system.
    Handles storing and retrieving document embeddings using FAISS index.
    """
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of the embeddings (default 1536 for Qwen embeddings)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.documents: List[Dict[str, Any]] = []  # Store document metadata
        self.doc_id_to_index: Dict[str, int] = {}  # Map document IDs to FAISS indices
        
    def add_embeddings(self, embeddings: List[np.ndarray], documents: List[Dict[str, Any]]):
        """
        Add embeddings and corresponding documents to the store.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of document metadata dictionaries
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
            
        # Normalize embeddings for cosine similarity
        normalized_embeddings = []
        for emb in embeddings:
            if isinstance(emb, list):
                emb = np.array(emb)
            norm = np.linalg.norm(emb)
            if norm == 0:
                normalized_emb = emb
            else:
                normalized_emb = emb / norm
            normalized_embeddings.append(normalized_emb.astype('float32'))
        
        # Convert to matrix for FAISS
        embeddings_matrix = np.vstack(normalized_embeddings)
        
        # Add to FAISS index
        start_idx = len(self.documents)
        self.index.add(embeddings_matrix)
        
        # Store document metadata and create ID mapping
        for i, doc in enumerate(documents):
            doc_idx = start_idx + i
            self.documents.append(doc)
            # Use doc_id from document metadata, fallback to index
            doc_id = doc.get('doc_id', f'doc_{doc_idx}')
            self.doc_id_to_index[doc_id] = doc_idx
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document metadata and similarity scores
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_normalized = query_embedding
        else:
            query_normalized = query_embedding / query_norm
        
        query_matrix = query_normalized.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_matrix, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and idx != -1:
                results.append({
                    'document': self.documents[idx],
                    'similarity_score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        # Ensure parent directories exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'doc_id_to_index': self.doc_id_to_index,
            'dimension': self.dimension
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> 'FAISSVectorStore':
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to load FAISS index from
            metadata_path: Path to load metadata from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create vector store instance
        vector_store = cls(dimension=metadata['dimension'])
        vector_store.index = index
        vector_store.documents = metadata['documents']
        vector_store.doc_id_to_index = metadata['doc_id_to_index']
        
        return vector_store
    
    def get_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.dimension
    
    def get_document_count(self) -> int:
        """Return the number of stored documents."""
        return len(self.documents)


def get_or_load_faiss(
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl"
) -> Optional[FAISSVectorStore]:
    """
    Get cached FAISS vector store or load it if not cached.

    Args:
        index_path: Path to load the FAISS index from
        metadata_path: Path to load the metadata from

    Returns:
        Cached or loaded FAISSVectorStore instance or None if error
    """
    global _cached_vector_store

    # Return cached instance if available
    if _cached_vector_store is not None:
        return _cached_vector_store

    # Load and cache the vector store
    try:
        vector_store = load_faiss(index_path, metadata_path)
        if vector_store:
            _cached_vector_store = vector_store
            logging.info(f"Loaded and cached FAISS vector store with {vector_store.get_document_count()} documents")
        return vector_store
    except Exception as e:
        logging.error(f"Error loading FAISS vector store: {str(e)}")
        return None


def clear_faiss_cache():
    """Clear the cached FAISS vector store."""
    global _cached_vector_store
    _cached_vector_store = None
    logging.info("FAISS vector store cache cleared")


def process_and_save_to_faiss(document_path: str, 
                            index_path: str = "./vector_stores/faiss_index.bin", 
                            metadata_path: str = "./vector_stores/faiss_metadata.pkl",
                            type:str = 'txt') -> bool:
    """
    Process documents using document_processor, generate embeddings using embedding_processor,
    and save to FAISS vector store.
    
    Args:
        document_path: Path to directory with documents or specific document file
        index_path: Path to save the FAISS index
        metadata_path: Path to save the metadata
        
    Returns:
        Boolean indicating success
    """
    try:
        # Step 1: Process documents using document_processor
        print("Step 1: Processing documents...")
        if type == 'txt':
            chunks = process_documents_v2(document_path)
        elif type == 'pdf':
            chunks = process_documents_pdf(document_path)
        # print(f"Processed {len(chunks)} document chunks.\n")
        
        if not chunks:
            print("No documents to process. Exiting.")
            return False
        
        # Step 2: Generate embeddings for each chunk using embedding_processor
        print("Step 2: Generating embeddings...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            if i % 100 == 0:  # Print progress every 100 chunks
                print(f"Processing chunk {i+1}/{len(chunks)}")
            text_content = chunk['content'].page_content if hasattr(chunk['content'], 'page_content') else chunk['content']
            emb = embedding(text_content)  # Use your embedding function
            if emb is None:
                print(f"Warning: Failed to generate embedding for chunk {i+1}, skipping this chunk")
                continue
            embeddings.append(emb)
        
        # Step 3: Create vector store and add embeddings
        if not embeddings:
            print("No valid embeddings generated. Please check your API key and network connection.")
            return False

        print("Step 3: Creating FAISS vector store...\n")
        vector_store = FAISSVectorStore(dimension=len(embeddings[0]) if embeddings else 1536)
        vector_store.add_embeddings(embeddings, chunks)
        
        # Step 4: Save to disk
        print("Step 4: Saving to disk...\n")
        vector_store.save(index_path, metadata_path)
        
        print(f"Successfully saved {len(chunks)} documents to FAISS.")
        print(f"Index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return True
    except Exception as e:
        print(f"Error in processing and saving to FAISS: {str(e)}")
        return False


def save_embeddings_to_faiss(embeddings: List[List[float]], documents: List[Dict[str, Any]], 
                            index_path: str = "./vector_stores/faiss_index.bin", 
                            metadata_path: str = "./vector_stores/faiss_metadata.pkl") -> bool:
    """
    Convenience function to save pre-computed embeddings and documents to FAISS.
    
    Args:
        embeddings: List of embedding vectors
        documents: List of document metadata
        index_path: Path to save the FAISS index
        metadata_path: Path to save the metadata
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create vector store
        vector_store = FAISSVectorStore(dimension=len(embeddings[0]) if embeddings else 1536)
        
        # Add embeddings and documents
        vector_store.add_embeddings(embeddings, documents)
        
        # Save to disk
        vector_store.save(index_path, metadata_path)
        
        print(f"Successfully saved {len(documents)} documents to FAISS.")
        print(f"Index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return True
    except Exception as e:
        print(f"Error saving to FAISS: {str(e)}")
        return False


def load_faiss(index_path: str, metadata_path: str) -> Optional[FAISSVectorStore]:
    """
    Convenience function to load a FAISS vector store.
    
    Args:
        index_path: Path to load the FAISS index from
        metadata_path: Path to load the metadata from
        
    Returns:
        Loaded FAISSVectorStore instance or None if error
    """
    try:
        vector_store = FAISSVectorStore.load(index_path, metadata_path)
        print(f"Successfully loaded FAISS vector store with {vector_store.get_document_count()} documents.")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS vector store: {str(e)}")
        return None


def search_documents(query: str, index_path: str="./vector_stores/faiss_index.bin", metadata_path: str="./vector_stores/faiss_metadata.pkl", k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to search documents in FAISS using a query string.
    
    Args:
        query: Query string to search for
        index_path: Path to load the FAISS index from
        metadata_path: Path to load the metadata from
        k: Number of results to return
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        # Load vector store
        vector_store = load_faiss(index_path, metadata_path)
        if not vector_store:
            return []
        
        # Generate embedding for query
        query_embedding = embedding(query)
        
        # Perform search
        results = vector_store.search(query_embedding, k=k)
        
        return results
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        return []

def search_documents_v2(
    query: str,
    k: int
) -> list:
    """
    Search documents in FAISS using a query string.
    Returns structured results with document content and similarity scores.

    Args:
        query: Query string to search for
        k: Number of results to return

    Returns:
        List of dicts with 'raw_doc' and 'similarity' keys
    """
    index_path = "./vector_stores/faiss_index.bin"
    metadata_path = "./vector_stores/faiss_metadata.pkl"
    try:
        # Get cached or load vector store
        vector_store = get_or_load_faiss(index_path, metadata_path)
        if not vector_store:
            return [{"raw_doc": "无法加载向量存储，请确保向量数据库已正确创建。", "similarity": 0.0}]

        # Generate embedding for query
        query_embedding = embedding(query)
        # Perform search
        results = vector_store.search(query_embedding, k=k)

        # Format results as structured data
        formatted_results = []
        for result in results:
            # Extract content from Document object if needed
            content = result['document']['content']
            if hasattr(content, 'page_content'):
                content = content.page_content
            elif isinstance(content, dict):
                content = str(content)
            else:
                content = str(content)

            formatted_results.append({
                "raw_doc": content,
                "similarity": float(result['similarity_score'])
            })

        return formatted_results
    except Exception as e:
        logging.error(f"Error searching documents_v2: {str(e)}")
        return [{"raw_doc": f"搜索文档时发生错误: {str(e)}", "similarity": 0.0}]
    
# Example usage
if __name__ == "__main__":
    # Process documents and save to FAISS
    success = process_and_save_to_faiss(
        document_path="./docs/pdf_docs",  # Path to your documents
        index_path="./vector_stores/faiss_index.bin",
        metadata_path="./vector_stores/faiss_metadata.pkl",
        type="pdf"
    )
    
    if success:
        print("Documents processed and saved to FAISS successfully!")
        
        # Example of searching
        results = search_documents(
            query="乌合之众具体指的是什么，如何产生的？",
            index_path="./vector_stores/faiss_index.bin",
            metadata_path="./vector_stores/faiss_metadata.pkl",
            k=5
        )
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}: Score={result['similarity_score']:.4f}, Content='{result['document']['content'][:10]}...'")
    else:
        print("Failed to process documents.")