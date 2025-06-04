# illuminy_streamlit.py
import os
import uuid
import tempfile
from dotenv import load_dotenv
import openai
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# === Load API Key ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Helper Functions ===

def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def create_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

def ask_general_question(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content

def academic_write_feedback(text):
    prompt = f"""
You are Illuminy, an expert academic writing coach.

1. Improve the student's writing.
2. Explain the improvements.
3. Suggest where to add citations.

Text:
\"\"\"
{text}
\"\"\"

---
1. Improved Text:
2. Explanation:
3. Citation Suggestions:
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and encouraging academic writing coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return response.choices[0].message.content

def check_plagiarism(chunks, input_text):
    embedder = OpenAIEmbeddings()
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_vectors = embedder.embed_documents(chunk_texts)
    input_vector = embedder.embed_query(input_text)

    similarities = cosine_similarity([input_vector], chunk_vectors)[0]
    threshold = 0.75
    flagged = [(score, text) for score, text in zip(similarities, chunk_texts) if score >= threshold]
    return flagged

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Illuminy", layout="wide")
    st.title("📘 Illuminy Academic Assistant")

    mode = st.sidebar.radio("Select a tool:", [
        "General Chat",
        "Upload PDFs & Ask",
        "Writing Assistant",
        "Plagiarism Checker"
    ])

    if mode == "General Chat":
        question = st.text_area("Ask a general academic question:")
        if st.button("Ask"):
            if question.strip():
                st.write(ask_general_question(question))
            else:
                st.warning("Please enter a question.")

    elif mode == "Upload PDFs & Ask":
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            all_chunks = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uuid.uuid4()}.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                docs = load_pdf(temp_path)
                chunks = split_documents(docs)
                all_chunks.extend(chunks)
                os.remove(temp_path)

            vectorstore = build_vector_store(all_chunks)
            st.session_state.qa = create_qa_chain(vectorstore)
            st.session_state.chunks = all_chunks
            st.success("PDFs loaded and processed successfully.")

        if "qa" in st.session_state:
            question = st.text_input("Ask about the uploaded PDFs:")
            if question:
                answer = st.session_state.qa.invoke({"query": question})
                st.write(answer["result"])

    elif mode == "Writing Assistant":
        user_text = st.text_area("Paste your paragraph:")
        if st.button("Improve Writing"):
            if user_text.strip():
                st.text_area("Illuminy's Help", academic_write_feedback(user_text), height=300)
            else:
                st.warning("Please enter some text.")

    elif mode == "Plagiarism Checker":
        if "chunks" not in st.session_state:
            st.warning("Please upload and process PDFs first in 'Upload PDFs & Ask' mode.")
        else:
            user_text = st.text_area("Text to check:")
            if st.button("Check Plagiarism"):
                matches = check_plagiarism(st.session_state.chunks, user_text)
                if matches:
                    for i, (score, excerpt) in enumerate(matches, 1):
                        st.markdown(f"### Match #{i} — Score: {score:.2f}")
                        st.code(excerpt[:500])
                else:
                    st.success("✅ No significant similarities found.")

if __name__ == "__main__":
    main()