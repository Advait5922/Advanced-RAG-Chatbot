# PDF Answering Using Mixtral 📋

## Overview

This project is a **Streamlit-based application** that enables users to upload PDF documents, process them into a searchable format, and ask questions based on the content. The application leverages **LlamaParse**, **Qdrant**, and **Mixtral AI** for document parsing, vector storage, and AI-powered question-answering resp. 

## Key Features
- **Upload and Parse PDF Files**: Users can upload PDFs, which are parsed into structured data for further analysis.
- **Semantic Search**: Query the content of uploaded PDFs using natural language questions.
- **Vector Database Integration**: Stores and manages embeddings using **Qdrant**.
- **Scalable and Optimized**: Automatically manages collections in the vector database to ensure efficient performance.
- **Reusable Resources**: Parsed data is cached to avoid redundant computations.

## Technologies Used
- **Streamlit**: For the user interface.
- **Mixtral AI (Groq)**: Open-source language model for answering queries.
- **Qdrant**: For vector storage and semantic search.
- **LlamaParse**: For parsing PDF content into embeddings.
- **FastEmbedEmbedding**: To generate vector embeddings for the parsed content.

---

## Prerequisites
1. **Python Environment**:
   - Python 3.8+
   - Required libraries are listed below.

2. **Secrets and API Keys**:
   Add the following keys to your Streamlit secrets (`.streamlit/secrets.toml`):
   ```toml
   LLAMAPARSE_API_KEY = "llamaparse_api_key"
   QDRANT_API_KEY = "qdrant_api_key"
   QDRANT_URL = "qdrant_instance_url"
   GROQ_API_KEY = "groq_api_key"
   ```

---

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. Install the required Python libraries:
   ```bash
   pip install streamlit nest-asyncio llama-index qdrant-client
   ```

3. Run the application:
   ```bash
   streamlit run RAG.py
   ```

---

## How It Works

1. **Upload PDF**:
   - Users upload a PDF document through the Streamlit interface.

2. **Parsing**:
   - The PDF is parsed into structured data using **LlamaParse**. Parsed data is cached locally using a hash-based system to prevent redundant processing.

3. **Vector Storage**:
   - Embeddings are generated using **FastEmbedEmbedding** and stored in a Qdrant collection. 
   - The application automatically manages Qdrant collections, ensuring no more than a specified maximum number of collections exist.

4. **Querying**:
   - Users can ask natural language questions based on the uploaded PDF content.
   - The question is processed using **Mixtral AI**, which queries the vector database to retrieve relevant answers.

---

## Key Functionalities

### 1. PDF Parsing and Storage
- Parses PDF documents into a structured format (Markdown).
- Caches parsed data locally for reusability.

### 2. Vector Database Management
- Uses Qdrant to store embeddings.
- Automatically creates and manages collections, removing the oldest collections when the limit is exceeded.

### 3. Semantic Search
- Provides a natural language search interface powered by Mixtral AI.
- Queries the vector database for relevant answers based on the uploaded document.

---

## Usage

1. **Upload a PDF**:
   - Drag and drop a PDF into the file uploader.

2. **Ask Questions**:
   - Enter a natural language question in the input box.
   - Click the **Search** button to retrieve answers based on the document.

---

## Customization

### Embedding Model
- The code currently uses **FastEmbedEmbedding** with the model `BAAI/bge-base-en-v1.5`.
- You can switch to other embedding models by modifying:
  ```python
  embed_model = FastEmbedEmbedding(model_name="your_model_name")
  ```

### LLM Model
- The code uses **Mixtral AI (Groq)**.
- You can replace it with another model by updating:
  ```python
  llm = Groq(model="model_name", api_key=GROQ_API_KEY)
  ```

### Qdrant Collection Limit
- Adjust the maximum number of collections by modifying:
  ```python
  manage_qdrant_collections(client, max_collections=5)
  ```

---

## Limitations
- **Dependency on API Keys**: Ensure that valid API keys are provided for LlamaParse, Qdrant, and Groq services.
- **PDF Content Quality**: Performance may vary based on the complexity and quality of the uploaded PDF content.
- **Resource Intensity**: Handling large documents or queries may be computationally intensive.

---

## Future Improvements
- Add support for multi-file uploads and batch processing.
- Include more advanced query analytics.
- Integrate additional vector storage options.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
Special thanks to the developers of **LlamaParse**, **Qdrant**, **Groq**, and the Streamlit community for their amazing tools and support.

---
