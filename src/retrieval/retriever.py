"""
Advanced retrieval system with hybrid search and reranking
"""
from typing import List, Dict, Optional
from loguru import logger
from .vector_store import VectorStore
import re


class HybridRetriever:
    """
    Advanced retriever combining multiple search strategies
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        enable_rerank: bool = False
    ):
        """
        Initialize hybrid retriever

        Args:
            vector_store: Vector store instance
            top_k: Number of results to return
            enable_rerank: Whether to use reranking (requires Cohere API)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.enable_rerank = enable_rerank

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        strategy: str = "hybrid"
    ) -> List[Dict]:
        """
        Retrieve relevant documents

        Args:
            query: Search query
            top_k: Number of results (overrides default)
            filter_metadata: Metadata filters
            strategy: Retrieval strategy ("semantic", "hybrid", "keyword")

        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Retrieving documents for query: '{query}' (strategy={strategy})")

        if strategy == "semantic":
            results = self._semantic_retrieval(query, top_k, filter_metadata)
        elif strategy == "hybrid":
            results = self._hybrid_retrieval(query, top_k, filter_metadata)
        elif strategy == "keyword":
            results = self._keyword_retrieval(query, top_k, filter_metadata)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply reranking if enabled
        if self.enable_rerank and len(results) > 0:
            results = self._rerank_results(query, results)

        return results[:top_k]

    def _semantic_retrieval(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict]
    ) -> List[Dict]:
        """Pure semantic search using embeddings"""
        return self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Get more results for potential reranking
            filter_metadata=filter_metadata
        )

    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict]
    ) -> List[Dict]:
        """Hybrid search combining semantic and keyword matching"""
        # Extract potential keywords from query
        keywords = self._extract_keywords(query)

        return self.vector_store.hybrid_search(
            query=query,
            top_k=top_k * 2,
            keywords=keywords
        )

    def _keyword_retrieval(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict]
    ) -> List[Dict]:
        """Keyword-based retrieval with BM25-like scoring"""
        # Get candidates using semantic search
        candidates = self.vector_store.search(
            query=query,
            top_k=top_k * 3,
            filter_metadata=filter_metadata
        )

        # Extract query terms
        query_terms = self._extract_keywords(query)

        # Score based on keyword matching
        scored_results = []
        for result in candidates:
            content_lower = result['content'].lower()

            # Count keyword occurrences
            term_frequencies = {}
            for term in query_terms:
                term_lower = term.lower()
                count = content_lower.count(term_lower)
                if count > 0:
                    term_frequencies[term] = count

            # Calculate BM25-like score
            bm25_score = sum(term_frequencies.values())

            # Combine with semantic score
            combined_score = (
                result['similarity_score'] * 0.6 +
                min(bm25_score / 10.0, 1.0) * 0.4
            )

            result['combined_score'] = combined_score
            result['keyword_matches'] = term_frequencies
            scored_results.append(result)

        # Sort by combined score
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)

        return scored_results

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query
        Removes common words and keeps important terms
        """
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'our', 'my'
        }

        # Tokenize and clean
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stop words and short words
        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 2
        ]

        return keywords

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder or simple heuristics
        In production, this would use Cohere Rerank or a cross-encoder model
        """
        logger.info("Reranking results...")

        # Simple heuristic reranking (in production, use Cohere or cross-encoder)
        for result in results:
            # Boost score based on various factors
            boost = 1.0

            # Boost if query terms appear in metadata
            metadata = result['metadata']

            # Boost if document type matches query context
            doc_type = metadata.get('document_type', '').lower()
            query_lower = query.lower()

            if doc_type in query_lower:
                boost += 0.2

            # Boost if section name is relevant
            section = metadata.get('section', '').lower()
            if any(term in section for term in self._extract_keywords(query)):
                boost += 0.1

            # Apply boost
            result['similarity_score'] *= boost
            result['reranked'] = True

        # Re-sort by boosted scores
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results

    def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents considering conversation context

        Args:
            query: Current query
            conversation_history: Previous queries/messages
            top_k: Number of results

        Returns:
            Retrieved documents with context
        """
        if top_k is None:
            top_k = self.top_k

        # If we have conversation history, create an enriched query
        enriched_query = query

        if conversation_history:
            # Use last 2-3 messages for context
            recent_context = conversation_history[-3:]
            context_terms = []

            for msg in recent_context:
                context_terms.extend(self._extract_keywords(msg))

            # Add important context terms to query
            unique_context = list(set(context_terms))[:5]
            if unique_context:
                enriched_query = f"{query} {' '.join(unique_context)}"
                logger.info(f"Enriched query with context: {enriched_query}")

        # Retrieve with enriched query
        return self.retrieve(enriched_query, top_k=top_k)

    def multi_query_retrieval(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        dedup: bool = True
    ) -> List[Dict]:
        """
        Retrieve documents for multiple related queries and combine results

        Args:
            queries: List of search queries
            top_k: Total number of results to return
            dedup: Remove duplicate results

        Returns:
            Combined results from all queries
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Multi-query retrieval for {len(queries)} queries")

        all_results = []
        seen_ids = set()

        # Retrieve for each query
        for query in queries:
            results = self.retrieve(query, top_k=top_k)

            for result in results:
                result_id = result['id']

                # Add if not duplicate or dedup disabled
                if not dedup or result_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result_id)

        # Sort by score and limit
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return all_results[:top_k]


class CitationRetriever(HybridRetriever):
    """
    Retriever that adds citation information to results
    """

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        strategy: str = "hybrid"
    ) -> List[Dict]:
        """Override to add citation information"""
        results = super().retrieve(query, top_k, filter_metadata, strategy)

        # Add citation to each result
        for i, result in enumerate(results, 1):
            citation = self._generate_citation(result['metadata'], i)
            result['citation'] = citation

        return results

    def _generate_citation(self, metadata: Dict, index: int) -> str:
        """
        Generate a citation string for a result

        Args:
            metadata: Document metadata
            index: Result index

        Returns:
            Citation string
        """
        doc_type = metadata.get('document_type', 'Unknown')
        filename = metadata.get('filename', 'Unknown')
        section = metadata.get('section', '')

        citation = f"[{index}] {doc_type}"

        if section:
            # Clean section name
            section_clean = section.replace('\n', ' ').strip()[:100]
            citation += f" - {section_clean}"

        citation += f" (Source: {filename})"

        return citation


if __name__ == "__main__":
    # Test the retriever
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent.parent))

    from ingestion import DocumentLoader, RegulationChunker
    from retrieval.vector_store import VectorStore

    # Initialize
    vector_store = VectorStore(
        persist_directory="./data/chroma_db",
        collection_name="test_compliance"
    )

    retriever = CitationRetriever(vector_store, top_k=3)

    # Test queries
    queries = [
        "What are the data retention requirements?",
        "GDPR right to erasure",
        "HIPAA security safeguards"
    ]

    for query in queries:
        print(f"\n\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = retriever.retrieve(query)

        for result in results:
            print(f"\n{result['citation']}")
            print(f"Score: {result['similarity_score']:.4f}")
            print(f"Preview: {result['content'][:150]}...")
