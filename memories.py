import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class ConversationMemory:
    """
    A class for managing conversation memory using a vector store. 
    Provides methods to store, retrieve, and prune conversational memories.

    Attributes:
        embeddings: An embeddings object used for generating vector representations of text.
        max_memory_size: Maximum number of memories to retain in the store.
        memory_retention_days: Number of days to retain memories before considering them for pruning.
        memory_store: A Chroma vector store instance to manage memory storage and retrieval.
    """
    def __init__(
        self, 
        embeddings: Embeddings, 
        max_memory_size: int = 100, 
        memory_retention_days: int = 30
    ):
        """
        Initializes the ConversationMemory instance.

        Args:
            embeddings: An embeddings object for generating vector representations of text.
            max_memory_size: Maximum number of memories to retain in the store (default: 100).
            memory_retention_days: Number of days to retain memories (default: 30).
        """
        self.embeddings = embeddings
        self.max_memory_size = max_memory_size
        self.memory_retention_days = memory_retention_days
        
        # Create the memory store
        self.memory_store = self._create_memory_store()

    def _create_memory_store(self) -> Chroma:
        """
        Creates or resets the Chroma vector store for managing conversation memories.

        Returns:
            A Chroma vector store instance configured for conversation memory.
        """
        return Chroma(
            embedding_function=self.embeddings,
            collection_name="conversation_memory"
        )

    def add_memory(
        self, 
        question: str, 
        answer: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Adds a new memory (question-answer pair) to the memory store.

        Args:
            question: The user's query or question.
            answer: The response or answer to the question.
            metadata: Optional additional metadata to associate with the memory.

        Steps:
            1. Generate a unique memory ID.
            2. Create default metadata, including the timestamp and memory ID.
            3. Merge any additional metadata provided.
            4. Create a `Document` representing the memory and add it to the memory store.
            5. Prune the memory store to maintain its size within limits.
        """
        # Step 1: Generate a unique memory ID
        memory_id = str(uuid.uuid4())
        
        # Step 2: Define default metadata
        default_metadata = {
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id
        }
        
        # Step 3: Update with provided metadata
        if metadata:
            default_metadata.update(metadata)
        
        # Step 4: Create a Document for the memory
        memory_doc = Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata=default_metadata
        )
        
        # Add the memory to the store
        self.memory_store.add_documents([memory_doc])
        
        # Step 5: Prune older memories if needed
        self._prune_memories()

    def retrieve_relevant_memories(
        self, 
        query: str, 
        top_k: int = 3
    ) -> List[Document]:
        """
        Retrieves the most relevant memories for a given query.

        Args:
            query: The input query to search for relevant memories.
            top_k: The maximum number of relevant memories to retrieve (default: 3).

        Returns:
            A list of the top-k relevant `Document` objects, or an empty list if no memories are available.

        Steps:
            1. Check if the memory store has any documents.
            2. Adjust `top_k` to ensure it does not exceed the total number of memories.
            3. Perform a similarity search to retrieve relevant memories.
        """
        # Step 1: Check if the memory store is empty
        if self.memory_store._collection.count() == 0:
            return []
        
        # Step 2: Adjust the number of memories to retrieve
        safe_k = min(top_k, max(1, self.memory_store._collection.count()))
        
        try:
            # Step 3: Perform similarity search
            return self.memory_store.similarity_search(query, k=safe_k)
        except Exception:
            return []

    def _prune_memories(self) -> None:
        """
        Prunes the memory store to ensure it stays within the maximum memory size.

        Steps:
            1. Check if the memory store is empty.
            2. Retrieve all existing memories from the store.
            3. Sort memories by timestamp in descending order (newest first).
            4. Keep only the top `max_memory_size` memories.
            5. Reinitialize the memory store and add the retained memories.
        """
        # Step 1: Check if the memory store is empty
        if self.memory_store._collection.count() == 0:
            return
        
        # Step 2: Retrieve all existing memories (limit retrieval to 1000 for safety)
        total_memories = self.memory_store._collection.count()
        safe_retrieval_count = min(total_memories, 1000)
        existing_memories = self.memory_store.similarity_search("", k=safe_retrieval_count)
        
        # Step 3: Sort memories by timestamp (newest first)
        sorted_memories = sorted(
            existing_memories,
            key=lambda x: datetime.fromisoformat(
                x.metadata.get('timestamp', datetime.min.isoformat())
            ),
            reverse=True
        )
        
        # Step 4: Retain only the top `max_memory_size` memories
        memories_to_keep = sorted_memories[:self.max_memory_size]
        
        # Step 5: Reinitialize the memory store and add retained memories
        self.memory_store = self._create_memory_store()
        if memories_to_keep:
            self.memory_store.add_documents(memories_to_keep)
