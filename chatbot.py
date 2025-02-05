from typing import List, Dict, Any, Optional
from datetime import datetime
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from query_processor import AdvancedQueryProcessor
from query_retriever import MultiQueryRetriever, State
from memories import ConversationMemory

class AdvancedQAChatbot:
    """
    A chatbot designed for advanced question-answering using query reformulation, 
    multi-query retrieval, and response generation.

    Attributes:
        llm: A language model for generating responses.
        embeddings: An embeddings object for vectorization of text.
        vector_store: A vector store for retrieving documents.
        documents: A list of documents for contextual search.
        max_history_length: Maximum number of conversation history entries to retain.
        confidence_threshold: Minimum confidence score for valid responses.
        query_processor: An instance of `AdvancedQueryProcessor` for query reformulation.
        multi_query_retriever: Retrieves relevant documents for queries.
        prompt_template: A prompt template for response generation.
        graph: A `StateGraph` for managing state transitions in the chatbot.
    """
    def __init__(self, 
                 llm: BaseLanguageModel,
                 embeddings: Embeddings, 
                 vector_store: VectorStore, 
                 documents: List[Document]):
        """
        Initializes the chatbot with the given language model, embeddings, 
        vector store, and documents.

        Args:
            llm: A language model for generating responses.
            embeddings: An embeddings object for creating vector representations of text.
            vector_store: A vector store instance for document retrieval.
            documents: A list of documents to use as the knowledge base.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.documents = documents
        self.max_history_length = 5
        self.confidence_threshold = 0.3
        
        # Setup query processor with default single-query behavior
        self.query_processor = AdvancedQueryProcessor(self.llm, num_queries=1)
        
        # Initialize the MultiQueryRetriever
        self.multi_query_retriever = MultiQueryRetriever(
            self.vector_store, 
            self.documents, 
            self.query_processor,
            k=3,
            fetch_k=10,
            lambda_mult=0.5
        )
        
        # Define the response prompt template
        self.prompt_template = PromptTemplate.from_template(
            "Use the following pieces of context to answer the question at the end.\n"
            "If you don't know the answer, just say this exact sentence and stop: "
            "\"Sorry, I didn't understand your question. Do you want to connect with a live agent?\""
            "Don't try to make up an answer or add any extra text to the answer.\n"
            "Always say \"Is there anything else I can help you with?\" at the end of the answer, only if you did find an answer.\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Helpful Answer:"
        )
        
        # Create the state graph
        self.graph = self._create_graph()

    def _weighted_history_context(self, history: List[Dict[str, str]], current_query: str) -> str:
        """
        Generates a weighted conversation history context using exponentially 
        decreasing weights for older entries.

        Args:
            history: A list of conversation entries (role and content).
            current_query: The current user query.

        Returns:
            A string combining weighted history and the current query.
        """
        weighted_history = []
        for i, entry in enumerate(reversed(history)):
            weight = math.exp(-0.5 * i)  # Exponentially decreasing weight
            weighted_history.append(f"[Weight:{weight:.2f}] {entry['role']}: {entry['content']}")
        return " ".join(weighted_history) + " " + current_query

    def _check_response_confidence(self, 
                                   context: List[Document], 
                                   question: str, 
                                   answer: str) -> float:
        """
        Calculates the confidence score of a response using semantic similarity 
        and token overlap metrics.

        Args:
            context: List of documents providing context for the query.
            question: The user-provided query string.
            answer: The generated answer string.

        Returns:
            A confidence score (float) between 0 and 1.
        """
        try:
            # Semantic similarity between context and answer
            context_text = " ".join([doc.page_content for doc in context])
            context_embedding = self.embeddings.embed_query(context_text)
            answer_embedding = self.embeddings.embed_query(answer)
            
            semantic_similarity = cosine_similarity(
                np.array(context_embedding).reshape(1, -1),
                np.array(answer_embedding).reshape(1, -1)
            )[0][0]
            
            # Token overlap metric
            context_tokens = set(context_text.lower().split())
            answer_tokens = set(answer.lower().split())
            token_overlap = len(context_tokens.intersection(answer_tokens)) / len(context_tokens)
            
            # Combined confidence score
            confidence = (semantic_similarity + token_overlap) / 2
            
            return confidence
        except Exception:
            return 0.5  # Default confidence if calculation fails

    def _create_graph(self) -> StateGraph:
        """
        Builds the state graph for retrieval and response generation.

        Returns:
            A compiled `StateGraph` object.
        """
        def retrieve(state: State):
            # Combine history and question for retrieval
            query_with_history = " ".join([
                f"{entry['role']}: {entry['content']}" 
                for entry in state.get("history", [])
            ] + [state['question']])
            retrieved_docs = self.multi_query_retriever.retrieve(query_with_history)
            return {"context": retrieved_docs}
        
        def generate(state: State):
            # Generate a response using context and conversation history
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt_template.invoke({
                "question": state["question"], 
                "context": docs_content + "\n\nPrevious Conversation Context:\n" +
                "\n".join([f"{entry['role']}: {entry['content']}" for entry in state.get('history', [])])
            })
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        
        # Build and compile the state graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

    def query(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Processes a user query and generates a response using the chatbot's state graph.

        Args:
            question: The user's question string.
            thread_id: Identifier for the conversation thread (default: "default").

        Returns:
            A dictionary containing the response, confidence, and context.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Retrieve the previous state and history
            previous_state = self.graph.get_state(config)
            chat_history = previous_state.get('history', [])
        except Exception:
            chat_history = []

        # Update chat history with the new question
        chat_history.append({"role": "user", "content": question})
        chat_history = chat_history[-self.max_history_length:]

        # Invoke the graph for response generation
        enhanced_context = self._weighted_history_context(chat_history, question)
        response = self.graph.invoke({
            "question": question, 
            "chat_history": chat_history, 
            "enchanced_context": enhanced_context
        }, config=config)

        # Validate the response confidence
        confidence = self._check_response_confidence(
            response.get('context', []), 
            question, 
            response['answer']
        )
        
        if confidence < self.confidence_threshold:
            response['answer'] = (
                "Sorry, I didn't understand your question. "
                "Do you want me to connect with a live agent?"
            )
            response['confidence'] = confidence
        
        # Update the conversation history
        chat_history.append({"role": "assistant", "content": response['answer']})
        return response

    def reformulate_query(self, query: str) -> str:
        """
        Reformulates a query for improved retrieval effectiveness.

        Args:
            query: The original query string.

        Returns:
            Reformulated query string.
        """
        return self.query_processor.reformulate_query(query)
    
    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generates variations of a query to enhance search comprehensiveness.

        Args:
            query: The original query string.

        Returns:
            A list of query variations.
        """
        return self.query_processor.generate_queries(query)

class AdvancedQAChatbotMemory:
    """
    A chatbot designed to handle complex question-answering scenarios by utilizing
    advanced query processing, multi-query retrieval, and memory management.

    Attributes:
        llm: A language model for generating responses.
        embeddings: Embeddings object for vector representation of text.
        vector_store: A vector store for document retrieval.
        documents: A list of documents to use for answering questions.
        max_history_length: Maximum length of conversation history to retain.
        confidence_threshold: Minimum confidence score for a valid response.
        query_processor: Processes and reformulates user queries.
        multi_query_retriever: Retrieves relevant documents for queries.
        conversation_memory: Stores and retrieves past conversation memories.
        prompt_template: Template for generating chatbot responses.
        graph: State graph for managing retrieval and response generation.
    """
    def __init__(
        self, 
        llm: BaseLanguageModel, 
        embeddings: Embeddings, 
        vector_store: VectorStore, 
        documents: List[Document],
        max_history_length: int = 5,
        confidence_threshold: float = 0.3,
        num_queries: int = 1
    ):
        """
        Initializes the chatbot with the given components and configurations.

        Args:
            llm: Language model for generating answers.
            embeddings: Embeddings object for vectorization.
            vector_store: Vector store for document retrieval.
            documents: List of documents for context.
            max_history_length: Max number of history entries to retain.
            confidence_threshold: Threshold for determining response confidence.
            num_queries: Number of query variations to generate for retrieval.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.documents = documents
        self.max_history_length = max_history_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize query processor and multi-query retriever
        self.query_processor = AdvancedQueryProcessor(self.llm, num_queries)
        self.multi_query_retriever = MultiQueryRetriever(
            self.vector_store, 
            self.documents, 
            self.query_processor,
            k=3,
            fetch_k=10,
            lambda_mult=0.5
        )
        
        # Create the response prompt template and conversation memory
        self.prompt_template = self._create_prompt_template()
        self.conversation_memory = ConversationMemory(embeddings)
        
        # Build the state graph
        self.graph = self._create_graph()

    def _create_prompt_template(self) -> PromptTemplate:
        """
        Creates a prompt template for generating chatbot responses.

        Returns:
            A `PromptTemplate` object with placeholders for context, question, and memory.
        """
        return PromptTemplate.from_template(
            "Use the following pieces of context to answer the question at the end.\n"
            "Consider the following relevant conversation memories if they help answer the question or if the question is about the past interaction:\n"
            "Memory Context: {memory_context}\n"
            "If you don't know the answer, just say this exact sentence and stop: "
            "\"Sorry, I didn't understand your question. Do you want to connect with a live agent?\""
            "Don't try to make up an answer or add any extra text to the answer.\n"
            "Always say \"Is there anything else I can help you with?\" at the end of the answer, only if you did find an answer.\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Helpful Answer:"
        )

    def _weighted_history_context(
        self, 
        history: List[Dict[str, str]], 
        current_query: str
    ) -> str:
        """
        Generates a weighted history context by assigning exponentially decreasing
        weights to previous conversation turns.

        Args:
            history: A list of conversation entries (role and content).
            current_query: The current user query.

        Returns:
            A string combining the weighted history and current query.
        """
        weighted_history = []
        for i, entry in enumerate(reversed(history)):
            weight = math.exp(-0.5 * i)  # Exponentially decreasing weights
            weighted_history.append(
                f"[Weight:{weight:.2f}] {entry['role']}: {entry['content']}"
            )
        return " ".join(weighted_history) + f" Current Query: {current_query}"

    def _check_response_confidence(
        self, 
        context: List[Document], 
        question: str, 
        answer: str
    ) -> float:
        """
        Calculates the confidence score of the chatbot's response based on 
        semantic similarity and token overlap with the context.

        Args:
            context: List of relevant documents for the query.
            question: The user query.
            answer: The generated answer.

        Returns:
            A confidence score between 0 and 1.
        """
        try:
            # Compute semantic similarity between context and answer
            context_text = " ".join([doc.page_content for doc in context])
            context_embedding = self.embeddings.embed_query(context_text)
            answer_embedding = self.embeddings.embed_query(answer)
            semantic_similarity = cosine_similarity(
                np.array(context_embedding).reshape(1, -1),
                np.array(answer_embedding).reshape(1, -1)
            )[0][0]
            
            # Compute token overlap
            context_tokens = set(context_text.lower().split())
            answer_tokens = set(answer.lower().split())
            token_overlap = len(context_tokens.intersection(answer_tokens)) / len(context_tokens)
            
            return (semantic_similarity + token_overlap) / 2
        except Exception:
            return 0.5  # Default confidence if calculation fails

    def _create_graph(self) -> StateGraph:
        """
        Builds the state graph for managing retrieval and response generation.

        Returns:
            A compiled `StateGraph` object.
        """
        def retrieve(state: State):
            query_with_history = " ".join([
                f"{entry['role']}: {entry['content']}" 
                for entry in state.get("history", [])
            ] + [state['question']])
            
            retrieved_docs = self.multi_query_retriever.retrieve(query_with_history)
            relevant_memories = self.conversation_memory.retrieve_relevant_memories(query_with_history)
            
            return {"context": retrieved_docs + relevant_memories}
        
        def generate(state: State):
            memory_context = "\n".join([
                f"Memory {i+1}: {mem.page_content}" 
                for i, mem in enumerate(state.get("memory", []))
            ])
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            
            messages = self.prompt_template.invoke({
                "question": state["question"], 
                "context": docs_content,
                "memory_context": memory_context,
                "history": "\n".join([
                    f"{entry['role']}: {entry['content']}" 
                    for entry in state.get('history', [])
                ])
            })
            
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

    def query(
        self, 
        question: str, 
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Handles user queries by retrieving relevant context and generating responses.

        Args:
            question: The user query.
            thread_id: ID of the conversation thread (default: "default").

        Returns:
            A dictionary containing the chatbot's response and confidence score.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            previous_state = self.graph.get_state(config)
            chat_history = previous_state.get('history', []) if previous_state else []
        except Exception:
            chat_history = []

        # Update chat history
        chat_history.append({
            "role": "user", 
            "content": question, 
            "timestamp": datetime.now().isoformat()
        })
        chat_history = chat_history[-self.max_history_length:]

        # Retrieve response
        enhanced_context = self._weighted_history_context(chat_history, question)
        response = self.graph.invoke({
            "question": question, 
            "history": chat_history, 
            "enhanced_context": enhanced_context
        }, config=config)

        # Check response confidence
        confidence = self._check_response_confidence(
            response.get('context', []), 
            question, 
            response['answer']
        )
        
        if confidence < self.confidence_threshold:
            response['answer'] = (
                "Sorry, I didn't understand your question. "
                "Do you want me to connect with a live agent?"
            )
            response['confidence'] = confidence
        
        # Add memory and return response
        self.conversation_memory.add_memory(question, response['answer'])
        chat_history.append({"role": "assistant", "content": response['answer']})
        return response

    def reformulate_query(self, query: str) -> str:
        """
        Reformulates a user query to make it more precise and effective for retrieval.

        Args:
            query: The original user query.

        Returns:
            A reformulated query.
        """
        return self.query_processor.reformulate_query(query)
    
    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generates variations of a query to improve retrieval performance.

        Args:
            query: The original query.

        Returns:
            A list of query variations.
        """
        return self.query_processor.generate_queries(query)
