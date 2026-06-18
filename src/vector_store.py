import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class LocalVectorStore:
    """
    A local persistent vector store using FAISS and SentenceTransformers.
    Stores index and metadata locally under the specified directory.
    """
    def __init__(self, db_dir: str = ".edumind_db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_dir = db_dir
        self.model_name = model_name
        self.index_path = os.path.join(self.db_dir, "faiss.index")
        self.metadata_path = os.path.join(self.db_dir, "metadata.json")
        
        # Load SentenceTransformer model
        print("Loading local SentenceTransformer model...")
        self.model = SentenceTransformer(self.model_name)
        
        self.index = None
        self.chunks = []
        
        # Create database folder if it doesn't exist
        os.makedirs(self.db_dir, exist_ok=True)
        self.load()

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """
        Embeds chunks of text and adds them to the FAISS index and metadata storage.
        """
        if not new_chunks:
            return
            
        texts = [chunk["text"] for chunk in new_chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        dimension = embeddings.shape[1]
        
        # If the index does not exist, initialize flat L2 index
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
            
        # Add to index
        self.index.add(embeddings.astype("float32"))
        
        # Append metadata
        self.chunks.extend(new_chunks)
        
        # Save updates to disk
        self.save()
        print("Embeddings added and database updated successfully.")

    def query(self, prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the vector store and returns the top_k matching text chunks.
        """
        if self.index is None or not self.chunks:
            print("Vector store is empty. No index or documents loaded.")
            return []
            
        # Embed query
        query_emb = self.model.encode([prompt], convert_to_numpy=True, show_progress_bar=False)
        
        # Search index
        distances, indices = self.index.search(query_emb.astype("float32"), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                match = self.chunks[idx].copy()
                match["score"] = float(dist)
                results.append(match)
                
        return results

    def clear(self):
        """Clears index and metadata from disk and memory."""
        self.index = None
        self.chunks = []
        
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except Exception:
                pass
        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
            except Exception:
                pass
        print("Vector store database cleared.")

    def save(self):
        """Saves current state to local files."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self):
        """Loads state from local files if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                print(f"Loaded existing index containing {len(self.chunks)} chunks.")
            except Exception as e:
                print(f"Failed to load existing index: {e}. Starting fresh.")
                self.index = None
                self.chunks = []


if __name__ == "__main__":
    # Test script if executed directly
    store = LocalVectorStore()
    
    # Test chunks
    test_chunks = [
        {"text": "Python is a high-level programming language known for readability.", "metadata": {"source": "test.txt", "chunk": 0}},
        {"text": "FAISS is a library for efficient similarity search of dense vectors.", "metadata": {"source": "test.txt", "chunk": 1}},
        {"text": "Streamlit is an open-source framework to build beautiful web apps.", "metadata": {"source": "test.txt", "chunk": 2}},
    ]
    
    store.add_chunks(test_chunks)
    
    # Test query
    q = "tell me about streamlit"
    matches = store.query(q, top_k=2)
    print(f"\nQuery: {q}")
    for m in matches:
        print(f"- Match: '{m['text']}' (Source: {m['metadata']['source']}, Score: {m['score']:.4f})")
    
    # Clean up test
    store.clear()
