# RAG based PDF-Query Assistant

## Overview

The **PDF Query Assistant** is an advanced chatbot designed to help users query and extract information from PDF documents. It leverages state-of-the-art natural language processing (NLP) techniques, including query reformulation, multi-query retrieval, and response generation, to provide precise and context-aware answers to user questions. The chatbot is built using the **LangChain** framework, integrates **LangGraph** for stateful workflow management, and uses **Groq** for language model capabilities and **Hugging Face** for text embeddings.

This project is ideal for users who need to interact with large PDF documents, such as research papers, manuals, or reports, and want to extract relevant information quickly and efficiently.

---

## Features

1. **Query Reformulation**:
   - Automatically reformulates user queries to make them more precise and suitable for academic/institutional search.
   - Ensures that the reformulated query retains the original intent while improving clarity and searchability.

2. **Multi-Query Retrieval**:
   - Generates multiple query variations to enhance search comprehensiveness.
   - Combines semantic search (using vector embeddings) and BM25-based retrieval for improved document retrieval performance.

3. **Advanced Response Generation**:
   - Uses a language model to generate context-aware responses based on retrieved documents.
   - Includes confidence scoring to ensure the reliability of responses.

4. **Stateful Workflow with LangGraph**:
   - Utilizes **LangGraph** to create stateful workflows for managing retrieval and response generation processes.
   - Enhances the efficiency and modularity of the RAG (Retrieval-Augmented Generation) pipeline.

5. **Conversation Memory**:
   - Maintains a memory of past interactions to provide context-aware responses.
   - Automatically prunes older memories to stay within size limits.

6. **Streamlit Web Interface**:
   - Provides an intuitive and user-friendly interface for uploading PDFs and querying them.
   - Supports dark mode for better readability.

7. **Customizable Configuration**:
   - Allows users to configure the number of query variations, retrieval parameters, and response generation settings.

---

## Installation

### Steps

1. **Set Up the Environment**:
   - Creating a new virtual environment:
     Windows:
     ```bash
     python -m venv ragenv
     ```
   - Activate the virtual environment:
     ```bash
     ragenv\Scripts\activate
     ```

3. **Set Up API Keys**:
   - Create a `.env` file in the root directory and add your Groq API key:
     ```bash
     GROQ_API_KEY=your_groq_api_key_here
     ```

4. **Run the Application**:
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```

---

## Usage

1. **Upload a PDF**:
   - Use the sidebar to upload a PDF document. The chatbot will process the document and prepare it for querying.

2. **Enter a Query**:
   - Type your question in the query input box. The chatbot will reformulate your query and retrieve relevant information from the document.

3. **View Results**:
   - The chatbot will display the reformulated query and provide a response based on the retrieved information.
   - If the chatbot is unsure about the answer, it will prompt you to connect with a live agent.

4. **Conversation History**:
   - The chatbot maintains a history of your queries and responses, allowing for context-aware interactions.

---

## Code Structure

The project is organized into the following modules:

1. **`query_processor.py`**:
   - Contains the `AdvancedQueryProcessor` class, which handles query reformulation and generation of query variations.

2. **`query_retriever.py`**:
   - Implements the `MultiQueryRetriever` class, which combines semantic search and BM25-based retrieval for document retrieval.

3. **`memories.py`**:
   - Manages conversation memory using a vector store. Includes methods for adding, retrieving, and pruning memories.

4. **`chatbot.py`**:
   - Defines the `AdvancedQAChatbot` class, which integrates query processing, retrieval, and response generation.
   - Uses **LangGraph** to create stateful workflows for managing the RAG pipeline.

5. **`app.py`**:
   - The main Streamlit application that provides the user interface for interacting with the chatbot.

---

## Key Libraries and Their Roles

1. **LangChain**:
   - Provides the core framework for building language model applications.
   - Used for query reformulation, retrieval, and response generation.

2. **LangGraph**:
   - Manages stateful workflows within the RAG pipeline.
   - Enhances the modularity and efficiency of the retrieval and response generation processes.

3. **Groq**:
   - Powers the language model for generating responses and reformulating queries.

4. **Hugging Face**:
   - Provides text embeddings for semantic search and document retrieval.

5. **Chroma**:
   - Acts as the vector store for storing and retrieving document embeddings.

6. **PyMuPDF**:
   - Used for loading and processing PDF documents.

7. **Streamlit**:
   - Provides the web interface for interacting with the chatbot.

---

## Configuration

The chatbot can be customized using the following parameters:

- **Number of Query Variations**: Adjust the number of query variations generated for each query.
- **Retrieval Parameters**: Configure the number of documents to retrieve (`k`) and the balance between relevance and diversity (`lambda_mult`).
- **Confidence Threshold**: Set the minimum confidence score required for a response to be considered valid.

---

## Dependencies

The project relies on the following Python libraries:

- **LangChain**: Framework for building language model applications.
- **LangGraph**: For stateful workflow management.
- **Streamlit**: For building the web interface.
- **Groq**: For language model capabilities.
- **Hugging Face**: For text embeddings.
- **Chroma**: Vector store for document retrieval.
- **PyMuPDF**: For loading and processing PDF documents.

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LangChain** for providing the framework for building language model applications.
- **LangGraph** for enabling stateful workflow management.
- **Groq** for their powerful language models.
- **Hugging Face** for their open-source text embeddings.
- **Streamlit** for making it easy to build interactive web applications.

---

Enjoy using the **PDF Query Assistant**! 🚀
