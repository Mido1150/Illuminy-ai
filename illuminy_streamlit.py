import os
from dotenv import load_dotenv
import openai
import streamlit as st
import uuid
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def ask_illuminy_with_vectorstore(query, vectorstore):
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa.run(query)

# ---------- Streamlit App ----------
def main():
    st.title("📘 Illuminy Academic Assistant")
    st.markdown("Upload one or more PDF files and ask Illuminy questions.")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        # Give each uploaded file a unique temporary filename
        unique_filename = f"temp_{uuid.uuid4()}.pdf"
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and process
        docs = load_pdf(unique_filename)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)

        os.remove(unique_filename)  # Clean up temp file

    st.success(f"✅ Loaded {len(all_chunks)} chunks from {len(uploaded_files)} PDFs.")
    
    vectorstore = build_vector_store(all_chunks)

    query = st.text_input("💬 Ask Illuminy a question based on your documents:")
    if query:
        answer = ask_illuminy_with_vectorstore(query, vectorstore)
        st.markdown("### 🧠 Illuminy says:")
        st.write(answer)
    all_chunks = []
    for uploaded_file in uploaded_files:
        unique_filename = f"temp_{uuid.uuid4()}.pdf"
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_pdf(unique_filename)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)

        os.remove(unique_filename)  # optional cleanup

    st.success(f"✅ Loaded {len(all_chunks)} chunks from {len(uploaded_files)} PDFs.")
    vectorstore = build_vector_store(all_chunks)

    query = st.text_input("💬 Ask Illuminy a question based on your documents:")
    if query:
        answer = ask_illuminy_with_vectorstore(query, vectorstore)
        st.markdown("### 🧠 Illuminy says:")
        st.write(answer)

if __name__ == "__main__":
    main()