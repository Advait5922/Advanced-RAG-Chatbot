from rank_bm25 import BM25Okapi
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

class MultiQueryRetriever:
    """
    A retriever class that combines semantic search and BM25-based retrieval 
    for improved document retrieval performance.

    Attributes:
        vector_store: A vector-based search engine supporting Max Marginal Relevance (MMR).
        documents: A list of `Document` objects used as the retrieval corpus.
        query_processor: An object capable of reformulating queries and generating query variations.
        k: The number of top documents to retrieve in the final results.
        fetch_k: The number of candidate documents to fetch during MMR search.
        lambda_mult: The lambda parameter for MMR search, controlling the balance between relevance and diversity.
        semantic_weight: Weight assigned to semantic search scores in the final ranking (default: 0.7).
        bm25_weight: Weight assigned to BM25 scores in the final ranking (default: 0.3).
        bm25_corpus: A tokenized corpus for BM25 scoring.
        bm25: A BM25Okapi object for performing BM25-based retrieval.
    """

    def __init__(self, vector_store, documents: List[Document], query_processor, k: int = 3, fetch_k: int = 10, lambda_mult: float = 0.5):
        """
        Initializes the MultiQueryRetriever.

        Args:
            vector_store: A vector-based search engine instance.
            documents: A list of documents to use for retrieval.
            query_processor: An object capable of query reformulation and generating query variations.
            k: Number of top documents to retrieve (default: 3).
            fetch_k: Number of documents to fetch during semantic search (default: 10).
            lambda_mult: Lambda parameter for balancing relevance and diversity in MMR search (default: 0.5).

        Raises:
            ValueError: If the documents list is empty.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.vector_store = vector_store
        self.documents = documents

        # Tokenize the document content for BM25
        self.bm25_corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.bm25_corpus)

        self.query_processor = query_processor
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

        self.semantic_weight = 0.7
        self.bm25_weight = 0.3

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents for a given query using a combination of semantic 
        search and BM25-based retrieval.

        Args:
            query: The user-provided query string.

        Returns:
            A list of top `Document` objects based on combined retrieval scores.
        """
        # Reformulate the query
        reformulated_query = self.query_processor.reformulate_query(query)

        # Generate query variations if the query processor supports multiple queries
        query_variations = (
            self.query_processor.generate_queries(reformulated_query) 
            if self.query_processor.num_queries > 1 
            else [reformulated_query]
        )
        
        all_retrieved_docs = []
        for var_query in query_variations:
            # Perform semantic search
            semantic_docs = self.vector_store.max_marginal_relevance_search(
                query=var_query, 
                k=self.k,           
                fetch_k=self.fetch_k, 
                lambda_mult=self.lambda_mult 
            )

            # Perform BM25-based retrieval
            bm25_scores = self.bm25.get_scores(var_query.split())
            bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
            bm25_docs = [self.documents[i] for i in bm25_top_indices]
            
            all_retrieved_docs.extend(semantic_docs + bm25_docs)

        # Remove duplicates based on document content
        unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
        
        # Re-rank the documents using semantic search
        final_docs = self.vector_store.max_marginal_relevance_search(
            query=query, 
            k=self.k,
            fetch_k=len(unique_docs),
            lambda_mult=self.lambda_mult,
            documents=unique_docs
        )
        return final_docs


class State(TypedDict):
    """
    Represents the state of a conversational retrieval system.

    Attributes:
        question: The current user query.
        context: A list of retrieved `Document` objects providing context.
        answer: The generated answer to the user's query.
        history: A list of previous conversation turns, each represented as a dictionary.
        thread_id: A unique identifier for the conversation thread.
    """
    question: str 
    context: List[Document]
    answer: str
    history: List[dict]
    thread_id: str
