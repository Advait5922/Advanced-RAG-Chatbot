import os
import nest_asyncio
import streamlit as st
import pickle
import time
import json
import shutil
import hashlib
from datetime import datetime
from llama_parse import LlamaParse
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import qdrant_client
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

nest_asyncio.apply()

# Setting up secrets
LLAMAPARSE_API_KEY = st.secrets["LLAMAPARSE_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Setting environment variables
os.environ['QDRANT_URL'] = QDRANT_URL
os.environ['QDRANT_API_KEY'] = QDRANT_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

st.title("PDF answering using Mixtral 📋")

# Uploading PDF file
uploaded_file = st.file_uploader("Upload a text file", type="pdf")

@st.cache_resource
def initialize_resources():
    # Load or parse data function
    def load_or_parse_data(file_path):
        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
        # Using hash of file content to generate a unique file name
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        data_file = os.path.join(data_dir, f"parsed_data_{file_hash}.pkl")

        if os.path.exists(data_file):
            with open(data_file, "rb") as f:
                parsed_data = pickle.load(f)
        else:
            llama_parse_documents = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown").load_data([file_path])

            with open(data_file, "wb") as f:
                pickle.dump(llama_parse_documents, f)

            parsed_data = llama_parse_documents

        return parsed_data

    with st.spinner('Processing...'):
        # Clearing existing data directories
        data_dir = "./data"
        storage_dir = "./storage"
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        
        # Saving the uploaded file locally
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        save_path = os.path.join(data_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Loading secrets
        LLAMAPARSE_API_KEY = st.secrets["LLAMAPARSE_API_KEY"]
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
        QDRANT_URL = st.secrets["QDRANT_URL"]
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

        os.environ['QDRANT_URL'] = QDRANT_URL
        os.environ['QDRANT_API_KEY'] = QDRANT_API_KEY
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY

        # Loading parsed data
        parsed_data = load_or_parse_data(save_path)

        # Setting up embedding model
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
        #embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        from llama_index.core import Settings
        Settings.embed_model = embed_model

        # Setting up LLM model
        llm = Groq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)
        Settings.llm = llm

        # Setting up Qdrant client
        client = QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_URL)
        # Path to store collection metadata
        metadata_file = "collection_metadata.json"
        # Function to create a new Qdrant collection
        def create_qdrant_collection(client, collection_name):
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
                )
                # Record the creation time
                record_collection_metadata(collection_name)
            except UnexpectedResponse as e:
                st.warning(f"Failed to create collection: {e}")

        # Function to record collection metadata
        def record_collection_metadata(collection_name):
            metadata = load_collection_metadata()
            collection_count = metadata.get("collection_count", 0) + 1
            metadata["collection_count"] = collection_count
            metadata[collection_name] = {"created": datetime.now().isoformat(), "number": collection_count}
            save_collection_metadata(metadata)

        # Function to load collection metadata
        def load_collection_metadata():
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    return json.load(f)
            return {"collection_count": 0}

        # Function to save collection metadata
        def save_collection_metadata(metadata):
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

        # Function to delete collection metadata
        def delete_collection_metadata(collection_name):
            metadata = load_collection_metadata()
            if collection_name in metadata:
                del metadata[collection_name]
                save_collection_metadata(metadata)

        # Function to delete the oldest collection if the total number exceeds a limit
        def manage_qdrant_collections(client, max_collections=5):
            metadata = load_collection_metadata()
            # st.write(f"Current collections: {len(metadata) - 1}")  # -1 to exclude the collection_count entry
            # st.write(f"Collections metadata: {metadata}")
    
            if len(metadata) - 1 > max_collections:
                # Sort collections by number (oldest first)
                oldest_collection = sorted(metadata.items(), key=lambda x: x[1]['number'] if isinstance(x[1], dict) else float('inf'))[0][0]
                try:
                    client.delete_collection(collection_name=oldest_collection)
                    delete_collection_metadata(oldest_collection)
            # st.warning(f"Deleted oldest collection: {oldest_collection}")
                except UnexpectedResponse as e:
                    st.warning(f"Failed to delete collection: {e}")

        # Function to create vector store with retries
        def create_vector_store_with_retries(client, parsed_data, collection_name, max_retries=5, delay=5):
            retries = 0
            while retries < max_retries:
                try:
                    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(documents=parsed_data, storage_context=storage_context, show_progress=True)
                    index.storage_context.persist()
                    return index
                except (UnexpectedResponse, Exception) as e:
                    retries += 1
                    st.warning(f"Batch upload failed {retries} times. Retrying in {delay} seconds...")
                    time.sleep(delay)
            st.error("Failed to upload data to Qdrant after multiple retries.")
            return None
            # Managing Qdrant collections to ensure there are at most max_collections
        manage_qdrant_collections(client)

        # Creating a unique collection name
        collection_name = f"qdrant_rag_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}"

        # Creating a new Qdrant collection
        create_qdrant_collection(client, collection_name)

        # Creating a new vector store with retries
        index = create_vector_store_with_retries(client, parsed_data, collection_name)

        if index:
            # Creating a query engine for the index
            query_engine = index.as_query_engine()

            # Querying the engine
            #response = query_engine.query(question)

            #st.success('Done!')
            #st.write("### Answer")
            #st.write(response.response)  # Display only the response text
    
        return query_engine

# Initialize resources
query_engine = initialize_resources()
# Input for the question
question = st.text_input("Enter your question:")

if st.button("Search"):
    response = query_engine.query(question)
    st.write(response.response)
