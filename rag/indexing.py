"""Document indexing - creates and manages vector embeddings."""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./data/chroma_db"
TOP_K_RETRIEVAL = 8  # Increased to get more documents
SIMILARITY_THRESHOLD = 0.5  # Lowered threshold to be less restrictive


class VectorStore:
    """Manages vector embeddings and similarity search."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL
        )
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            if os.path.exists(CHROMA_PERSIST_DIR):
                self.vector_store = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing vector store with {self.vector_store._collection.count()} documents")
            else:
                print("No existing vector store found. Will create new one when documents are added.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
    
    def add_documents(self, documents: List[LangchainDocument]) -> bool:
        """Add documents to the vector store with batch processing and rate limiting."""
        try:
            if not documents:
                print("No documents to add")
                return False
            
            # ChromaDB batch size limit is around 5000 documents
            BATCH_SIZE = 1000  # Conservative batch size
            total_docs = len(documents)
            
            print(f"Processing {total_docs} documents in batches of {BATCH_SIZE}...")
            
            if self.vector_store is None:
                # Create vector store with first batch
                first_batch = documents[:BATCH_SIZE]
                print(f"Creating vector store with first batch of {len(first_batch)} documents...")
                
                self.vector_store = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_PERSIST_DIR
                )
                
                print(f"✅ Created vector store with {len(first_batch)} documents")
                
                # Process remaining documents in batches
                remaining_docs = documents[BATCH_SIZE:]
            else:
                remaining_docs = documents
            
            # Process remaining documents in batches
            for i in range(0, len(remaining_docs), BATCH_SIZE):
                batch = remaining_docs[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + (1 if self.vector_store else 2)
                
                print(f"Processing batch {batch_num}: {len(batch)} documents...")
                
                try:
                    self.vector_store.add_documents(batch)
                    print(f"✅ Added batch {batch_num} ({len(batch)} documents)")
                    
                    # Add delay between batches to avoid rate limiting
                    if i + BATCH_SIZE < len(remaining_docs):
                        import time
                        print("⏳ Waiting 2 seconds to avoid rate limits...")
                        time.sleep(2)
                        
                except Exception as batch_error:
                    print(f"❌ Error processing batch {batch_num}: {batch_error}")
                    
                    # If rate limited, wait and retry
                    if "rate_limit_exceeded" in str(batch_error):
                        import time
                        print("⏳ Rate limit hit, waiting 10 seconds...")
                        time.sleep(10)
                        
                        try:
                            self.vector_store.add_documents(batch)
                            print(f"✅ Retry successful for batch {batch_num}")
                        except Exception as retry_error:
                            print(f"❌ Retry failed for batch {batch_num}: {retry_error}")
                            return False
                    else:
                        return False
            
            print(f"🎉 Successfully processed all {total_docs} documents!")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = TOP_K_RETRIEVAL,
        filter_metadata: Optional[Dict] = None
    ) -> List[LangchainDocument]:
        """Perform similarity search."""
        if self.vector_store is None:
            print("Vector store not initialized")
            return []
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            # Be more lenient with similarity scores - return more documents
            # Lower scores are better in similarity search
            # Return documents with reasonable similarity (less restrictive filtering)
            filtered_docs = []
            for doc, score in docs_with_scores:
                # Only filter out documents with very high scores (very dissimilar)
                if score < 1.5:  # More lenient threshold
                    filtered_docs.append(doc)
            
            # If we filtered out too many, return the best ones anyway
            if len(filtered_docs) < max(3, k//2):
                filtered_docs = [doc for doc, score in docs_with_scores[:k]]
            
            return filtered_docs
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def get_relevant_documents(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        metadata_filter: Optional[Dict] = None
    ) -> List[LangchainDocument]:
        """Get relevant documents for a query."""
        return self.similarity_search(query, k, metadata_filter)
    
    def delete_collection(self):
        """Delete the entire vector store collection."""
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
                self.vector_store = None
                print("Vector store collection deleted")
                return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        if self.vector_store is None:
            return {"status": "not_initialized", "count": 0}
        
        try:
            count = self.vector_store._collection.count()
            return {
                "status": "active",
                "count": count,
                "persist_directory": CHROMA_PERSIST_DIR
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}
    
    def search_by_metadata(
        self, 
        metadata_filter: Dict[str, Any], 
        limit: int = 10
    ) -> List[LangchainDocument]:
        """Search documents by metadata filters."""
        if self.vector_store is None:
            return []
        
        try:
            # If no filter provided, just do a general search
            if not metadata_filter:
                results = self.vector_store.similarity_search(
                    query="",
                    k=limit
                )
            else:
                results = self.vector_store.similarity_search(
                    query="",
                    k=limit,
                    filter=metadata_filter
                )
            return results
        except Exception as e:
            print(f"Error searching by metadata: {e}")
            return []