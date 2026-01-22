"""Document retrieval - finds relevant documents for queries with advanced search."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document as LangchainDocument
from rank_bm25 import BM25Okapi
import numpy as np

from .indexing import VectorStore

# Configuration
RERANK_TOP_K = 3
ENABLE_RERANKING = True


class AdvancedRetriever:
    """Enhanced retriever with reranking and hybrid search capabilities."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25_index = None
        self.documents_corpus = []
    
    def build_bm25_index(self, documents: List[LangchainDocument]):
        """Build BM25 index for keyword search."""
        try:
            corpus = []
            self.documents_corpus = documents
            
            for doc in documents:
                tokens = doc.page_content.lower().split()
                corpus.append(tokens)
            
            self.bm25_index = BM25Okapi(corpus)
            print(f"Built BM25 index with {len(corpus)} documents")
            
        except Exception as e:
            print(f"Error building BM25 index: {e}")
    
    def bm25_search(self, query: str, k: int = 10) -> List[tuple]:
        """Perform BM25 keyword search."""
        if self.bm25_index is None:
            return []
        
        try:
            query_tokens = query.lower().split()
            scores = self.bm25_index.get_scores(query_tokens)
            
            top_indices = np.argsort(scores)[::-1][:k]
            results = [
                (self.documents_corpus[i], scores[i]) 
                for i in top_indices if scores[i] > 0
            ]
            
            return results
            
        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return []
    
    def hybrid_retrieve(
        self, 
        query: str, 
        k: int = 10,
        semantic_weight: float = 0.7,
        metadata_filter: Optional[Dict] = None
    ) -> List[LangchainDocument]:
        """Combine semantic and keyword search results."""
        try:
            semantic_docs = self.vector_store.similarity_search(
                query, k=k, filter_metadata=metadata_filter
            )
            
            bm25_results = self.bm25_search(query, k=k)
            
            if not bm25_results:
                return semantic_docs
            
            combined_docs = self._combine_search_results(
                semantic_docs, 
                bm25_results, 
                semantic_weight
            )
            
            return combined_docs[:k]
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return self.vector_store.similarity_search(query, k, metadata_filter)
    
    def _combine_search_results(
        self, 
        semantic_docs: List[LangchainDocument], 
        bm25_results: List[tuple],
        semantic_weight: float
    ) -> List[LangchainDocument]:
        """Combine and score results from different search methods."""
        doc_scores = {}
        
        # Score semantic results
        for i, doc in enumerate(semantic_docs):
            doc_id = doc.metadata.get('doc_id', str(i))
            semantic_score = 1.0 / (i + 1)
            doc_scores[doc_id] = {
                'doc': doc,
                'semantic_score': semantic_score,
                'bm25_score': 0.0
            }
        
        # Score BM25 results
        max_bm25_score = max([score for _, score in bm25_results]) if bm25_results else 1.0
        
        for doc, bm25_score in bm25_results:
            doc_id = doc.metadata.get('doc_id', 'unknown')
            normalized_bm25 = bm25_score / max_bm25_score if max_bm25_score > 0 else 0
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['bm25_score'] = normalized_bm25
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0.0,
                    'bm25_score': normalized_bm25
                }
        
        # Calculate combined scores
        for doc_id in doc_scores:
            semantic_score = doc_scores[doc_id]['semantic_score']
            bm25_score = doc_scores[doc_id]['bm25_score']
            
            combined_score = (
                semantic_weight * semantic_score + 
                (1 - semantic_weight) * bm25_score
            )
            doc_scores[doc_id]['combined_score'] = combined_score
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        return [item['doc'] for item in sorted_docs]
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[LangchainDocument], 
        top_k: int = RERANK_TOP_K
    ) -> List[LangchainDocument]:
        """Rerank documents based on query relevance."""
        if not ENABLE_RERANKING or len(documents) <= top_k:
            return documents[:top_k]
        
        try:
            scored_docs = []
            query_terms = set(query.lower().split())
            
            for doc in documents:
                content_terms = set(doc.page_content.lower().split())
                
                overlap = len(query_terms.intersection(content_terms))
                total_terms = len(query_terms)
                
                if total_terms > 0:
                    relevance_score = overlap / total_terms
                else:
                    relevance_score = 0
                
                scored_docs.append((doc, relevance_score))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in scored_docs[:top_k]]
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            return documents[:top_k]
    
    def retrieve_with_metadata_filter(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        k: int = 10,
        use_hybrid: bool = False
    ) -> List[LangchainDocument]:
        """Retrieve documents with metadata filtering."""
        try:
            if use_hybrid:
                docs = self.hybrid_retrieve(query, k=k*2)
                
                filtered_docs = []
                for doc in docs:
                    match = True
                    for key, value in metadata_filters.items():
                        if doc.metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                
                return filtered_docs[:k]
            else:
                return self.vector_store.similarity_search(
                    query, k=k, filter_metadata=metadata_filters
                )
                
        except Exception as e:
            print(f"Error in filtered retrieval: {e}")
            return []
    
    def get_diverse_results(
        self, 
        query: str, 
        k: int = 10,
        diversity_threshold: float = 0.8
    ) -> List[LangchainDocument]:
        """Get diverse results to avoid redundancy."""
        candidates = self.vector_store.similarity_search(query, k=k*2)
        
        if len(candidates) <= k:
            return candidates
        
        diverse_docs = [candidates[0]]
        
        for candidate in candidates[1:]:
            is_diverse = True
            candidate_words = set(candidate.page_content.lower().split())
            
            for selected_doc in diverse_docs:
                selected_words = set(selected_doc.page_content.lower().split())
                
                intersection = len(candidate_words.intersection(selected_words))
                union = len(candidate_words.union(selected_words))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > diversity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_docs.append(candidate)
                
            if len(diverse_docs) >= k:
                break
        
        return diverse_docs