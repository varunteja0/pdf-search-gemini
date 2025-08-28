# app.py
# Run with: streamlit run app.py
# Requirements: pip install streamlit pypdf sentence-transformers faiss-cpu google-generativeai watchdog numpy
# Get Gemini API key from https://aistudio.google.com/app/apikey

import streamlit as st
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
import tempfile
import datetime

# Set page config as the first Streamlit command
st.set_page_config(page_title="PDF Search with Gemini", page_icon="📄", layout="wide")

# Constants
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Fast and effective
LLM_MODEL = 'gemini-1.5-flash'  # Compatible Gemini model
MAX_CONTEXT_TOKENS = 8000  # Adjust based on Gemini model limits
CHUNK_SIZE = 500  # Words per chunk
INDEX_DIR = "index_storage"  # Directory to save indexes

# Create index storage directory
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

# Initialize embedding model
@st.cache_resource
def load_models():
    return SentenceTransformer(EMBEDDING_MODEL)

embedding_model = load_models()

# Function to extract and chunk text from PDF
def extract_and_chunk_pdf(pdf_path):
    chunks = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    words = text.split()
                    for i in range(0, len(words), CHUNK_SIZE):
                        chunk_text = ' '.join(words[i:i + CHUNK_SIZE])
                        chunks.append({
                            'page': page_num,
                            'text': chunk_text.strip()
                        })
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    return chunks

# Function to create and save FAISS index
def create_index(chunks, pdf_name):
    texts = [chunk['text'] for chunk in chunks if chunk['text']]
    if not texts:
        raise ValueError("No text extracted from PDF.")
    
    embeddings = embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    index_path = os.path.join(INDEX_DIR, f"{pdf_name}_{timestamp}.faiss")
    chunks_path = os.path.join(INDEX_DIR, f"{pdf_name}_{timestamp}_chunks.npy")
    faiss.write_index(index, index_path)
    np.save(chunks_path, chunks)
    
    return index, chunks, index_path, chunks_path

# Function to load existing index
def load_index(index_path, chunks_path):
    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True).tolist()
    return index, chunks

# Function to search index
def search(query, index, chunks, top_k=10):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and dist > 0.4:
            results.append(chunks[idx])
    return results

# Function to generate response using Gemini API
def generate_response(query, relevant_chunks, api_key=None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY environment variable or enter it in the sidebar.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(LLM_MODEL)
    
    context = "\n\n".join([f"Page {chunk['page']}: {chunk['text']}" for chunk in relevant_chunks])
    if len(context) > MAX_CONTEXT_TOKENS * 4:
        context = context[:MAX_CONTEXT_TOKENS * 4]
    
    prompt = f"""
You are a precise and helpful assistant specialized in answering queries based on PDF content.
Use only the provided context to answer. If the context doesn't have the information, say "I don't have enough information from the PDF to answer this."
Be concise, accurate, and cite page numbers where relevant.

Context:
{context}

Query: {query}

Response:
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'max_output_tokens': 1000,
                'temperature': 0.5,
                'top_p': 0.9,
            }
        )
        return response.text.strip()
    except Exception as e:
        raise ValueError(f"Gemini API error: {str(e)}")

# Streamlit App
st.title("📄 PDF Search & Query with Gemini")
st.markdown("""
Upload one or more PDFs, index their content, and ask natural language questions. 
Powered by semantic search and Google's Gemini API for accurate, context-aware responses.
""")

# Sidebar
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")
st.sidebar.markdown("**Note**: Ensure your Google Cloud project has billing enabled for extended API usage.")

# Initialize session state
if 'indexes' not in st.session_state:
    st.session_state.indexes = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# PDF uploader
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    try:
        for uploaded_file in uploaded_files:
            pdf_name = uploaded_file.name.split('.')[0]
            if pdf_name not in st.session_state.indexes:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name
                    
                    chunks = extract_and_chunk_pdf(pdf_path)
                    index, chunks, index_path, chunks_path = create_index(chunks, pdf_name)
                    st.session_state.indexes[pdf_name] = (index, chunks, index_path, chunks_path)
                    os.unlink(pdf_path)
                    st.success(f"Indexed {uploaded_file.name} ({len(chunks)} chunks).")
    
    except Exception as e:
        st.error(f"Error processing PDF(s): {str(e)}")

# Select PDF for querying
if st.session_state.indexes:
    pdf_name = st.selectbox("Select PDF to query", list(st.session_state.indexes.keys()))
    index, chunks, _, _ = st.session_state.indexes[pdf_name]
    
    # Query input
    query = st.text_input("Ask a question about the PDF:", key=f"query_{pdf_name}")
    
    if query:
        with st.spinner("Searching and generating response..."):
            try:
                relevant_chunks = search(query, index, chunks, top_k=10)
                if not relevant_chunks:
                    st.warning("No relevant content found in the PDF.")
                else:
                    response = generate_response(query, relevant_chunks, api_key)
                    st.markdown("### Response")
                    st.write(response)
                    
                    # Save to query history
                    st.session_state.query_history.append({
                        'pdf': pdf_name,
                        'query': query,
                        'response': response,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Show relevant chunks
                    with st.expander("View Relevant Chunks"):
                        for chunk in relevant_chunks:
                            st.write(f"**Page {chunk['page']}**: {chunk['text'][:300]}...")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Query history
if st.session_state.query_history:
    with st.expander("Query History"):
        for entry in reversed(st.session_state.query_history):
            st.markdown(f"**PDF**: {entry['pdf']} | **Time**: {entry['timestamp']}")
            st.write(f"**Query**: {entry['query']}")
            st.write(f"**Response**: {entry['response']}")
            st.markdown("---")

# Clear indexes button
if st.session_state.indexes and st.button("Clear All Indexes"):
    for _, (_, _, index_path, chunks_path) in st.session_state.indexes.items():
        if os.path.exists(index_path):
            os.unlink(index_path)
        if os.path.exists(chunks_path):
            os.unlink(chunks_path)
    st.session_state.indexes = {}
    st.success("All indexes cleared.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Sentence Transformers, FAISS, and Google's Gemini API. For production, consider using a vector database like Pinecone for scalability.")