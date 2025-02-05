from langchain_core.prompts import PromptTemplate
from difflib import SequenceMatcher

class AdvancedQueryProcessor:
    """
    A class for advanced query processing that includes query reformulation 
    and generating multiple query variations to enhance search efficiency.

    Attributes:
        llm: A language model object used for query reformulation and generation.
        num_queries: The number of query variations to generate.
    """
    def __init__(self, llm, num_queries: int):
        """
        Initializes the AdvancedQueryProcessor with a language model and a number of queries.

        Args:
            llm: A language model instance for processing queries.
            num_queries: The number of query variations to generate.
        """
        self.llm = llm
        self.num_queries = num_queries

    @staticmethod
    def is_similar(original: str, reformulated: str, threshold: float = 0.95) -> bool:
        """
        Determines whether the original query and the reformulated query are sufficiently similar.

        Args:
            original: The original query string.
            reformulated: The reformulated query string.
            threshold: Similarity threshold above which queries are considered similar (default: 0.95).

        Returns:
            True if the queries are similar, False otherwise.
        """
        # Calculate similarity using SequenceMatcher
        similarity = SequenceMatcher(None, original, reformulated).ratio()
        return similarity > threshold

    def reformulate_query(self, query: str) -> str:
        """
        Reformulates the input query to make it precise, clear, and suitable for academic/institutional search.

        Args:
            query: The original query string.

        Returns:
            Reformulated query string, or the original query if no reformulation is required or fails.

        Steps:
            1. Create a prompt for query reformulation using a template.
            2. Invoke the language model to reformulate the query.
            3. Check if the reformulated query is valid and distinct from the original.
            4. Return the reformulated query or the original query based on the similarity check.
        """
        # Step 1: Create a reformulation prompt
        reformulation_prompt = PromptTemplate.from_template(
            "You are an expert in information retrieval and query formulation. "
            "You are provided a query at the end. "
            "Reformulate the query to make it precise, clear, and searchable in an academic/institutional document. "
            "Remove irrelevant language. Avoid reformulation if the query is precise and grammatically correct. "
            "Only return the query and no other text. "
            "Original Query: {query}\n\n"
            "Reformulated Query:"
        )
        
        try:
            # Step 2: Use the language model to process the reformulation
            messages = reformulation_prompt.invoke({"query": query})
            response = self.llm.invoke(messages)
            reformulated_query = response.content.strip()
            
            # Step 3: Check similarity to the original query
            if not reformulated_query or self.is_similar(query, reformulated_query):
                return query
            
            # Step 4: Return the reformulated query if valid
            return reformulated_query
        except Exception as e:
            print(f"Error during query reformulation: {e}")
            return query

    def generate_queries(self, query: str) -> list:
        """
        Generates diverse variations of the input query to improve search comprehensiveness.

        Args:
            query: The original query string.

        Returns:
            A list of query variations.

        Steps:
            1. Create a prompt for generating query variations using a template.
            2. Invoke the language model to generate query variations.
            3. Parse and clean the query variations returned by the model.
            4. Return the variations or fallback options if the model fails.
        """
        # Step 1: Create a multi-query generation prompt
        multi_query_template = PromptTemplate.from_template(
            "You are an expert in generating diverse search query variations. "
            "Create {num_queries} different ways of expressing the same information to improve search comprehensiveness. "
            "Ensure each variation maintains the core intent of the original query and doesn't change the keywords. "
            "Only output the queries, each followed by a linebreak. "
            "Original Query: {query}\n\n"
            "Query Variations:\n"
        )
        
        try:
            # Step 2: Use the language model to generate query variations
            messages = multi_query_template.invoke({
                "query": query, 
                "num_queries": self.num_queries
            })
            response = self.llm.invoke(messages)
            
            # Step 3: Parse and clean the variations
            query_variations = response.content.split('\n')
            query_variations = [q.strip() for q in query_variations if q.strip()]
            
            # Step 4: Return the variations or fallback options
            if not query_variations:
                query_variations = [
                    query,
                    f"Alternative phrasing of: {query}",
                    f"More details about: {query}"
                ]
            
            return query_variations[:self.num_queries]
        except Exception as e:
            print(f"Error during query generation: {e}")
            return [
                query,
                f"Alternative phrasing of: {query}",
                f"More details about: {query}"
            ]
