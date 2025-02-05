import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chatbot import AdvancedQAChatbot

# Apply custom dark theme styling to the app
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
    }

    /* Input fields */
    .stTextInput>div>div>input {
        background-color: rgba(30, 30, 30, 0.7);
        color: #FFFFFF; /* Bright white for input text */
        border: 1px solid rgba(100, 100, 100, 0.2);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: rgba(100, 150, 255, 0.5);
        box-shadow: 0 6px 8px rgba(0,0,0,0.4);
        background-color: rgba(40, 40, 40, 0.9);
        color: #FFFFFF;
    }

    /* Text area for query input */
    .stTextArea>div>div>textarea {
        background-color: rgba(30, 30, 30, 0.7);
        color: #FFFFFF; /* Bright white for query input text */
        border: 1px solid rgba(100, 100, 100, 0.2);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: rgba(100, 150, 255, 0.5);
        box-shadow: 0 6px 8px rgba(0,0,0,0.4);
        background-color: rgba(40, 40, 40, 0.9);
        color: #FFFFFF;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4A6CF7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #6A8DFF;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    }

    /* Output box */
    .output-box {
        background-color: rgba(30, 30, 30, 0.8);
        color: #FFFFFF; /* Bright white for document query text */
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(100, 100, 100, 0.1);
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #4A6CF7 transparent #4A6CF7 transparent;
    }

    /* Additional text color adjustments */
    .stMarkdown {
        color: #E0E0E0;
    }
    .stMarkdown strong {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

def initialize_chatbot(pdf_file):
    """
    Initializes the chatbot by processing the uploaded PDF, generating document embeddings,
    and preparing the retrieval system.

    Args:
        pdf_file: The uploaded PDF file as a file-like object.

    Returns:
        An instance of `AdvancedQAChatbot` configured with the processed document data.
    """
    # Create a temporary directory for storing the uploaded file
    os.makedirs('temp', exist_ok=True)
    temp_path = os.path.join('temp', pdf_file.name)
    with open(temp_path, 'wb') as f:
        f.write(pdf_file.getvalue())
    
    # Initialize components
    llm = ChatGroq(model="llama3-70b-8192")  # Groq language model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Text embeddings
    vector_store = Chroma(embedding_function=embeddings)  # Vector store for document retrieval
    
    # Load and process the PDF
    loader = PyMuPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    
    # Create and return the chatbot instance
    chatbot = AdvancedQAChatbot(llm, embeddings, vector_store, all_splits)
    return chatbot

def main():
    """
    Main function for the Streamlit app. Manages the user interface and integrates
    chatbot functionalities for querying uploaded PDF documents.
    """
    st.title("PDF Query Assistant")
    
    with st.sidebar:
        st.header("Configuration")
        # Input field for API key
        groq_api_key = st.text_input("Groq API Key", type="password", 
                                     help="Your Groq API key for language model access")
        
        # PDF uploader
        uploaded_pdf = st.file_uploader("Upload PDF", type=['pdf'], 
                                        help="Select the PDF document to query")

    if uploaded_pdf is not None:
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
            return
        
        # Set the API key as an environment variable
        os.environ['GROQ_API_KEY'] = groq_api_key
        
        try:
            # Initialize chatbot
            chatbot = initialize_chatbot(uploaded_pdf)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return
        
        # Query input box
        query = st.text_area("Enter your query", 
                             placeholder="Ask a question about the uploaded PDF...",
                             height=100)

        if st.button("Query Document"):
            if query.strip():
                with st.spinner('Analyzing document...'):
                    # Generate reformulated query and retrieve the answer
                    reformulated_query = chatbot.reformulate_query(query)
                    response = chatbot.query(query)
                
                # Display reformulated query
                st.subheader("Query Analysis")
                st.markdown(f"**Reformulated Query:** *{reformulated_query}*", unsafe_allow_html=True)
                
                # Display response
                st.subheader("Response")
                st.markdown(f'<div class="output-box">{response["answer"]}</div>', 
                            unsafe_allow_html=True)
            else:
                st.warning("Please enter a query.")
    else:
        st.info("Upload a PDF in the sidebar to get started.")

if __name__ == "__main__":
    main()
