import os
from dotenv import load_dotenv
import openai
import streamlit as st
<<<<<<< HEAD
import uuid
=======
>>>>>>> 9df8fc6 (Initial commit for Illuminy Streamlit app)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
<<<<<<< HEAD
=======
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import tempfile
>>>>>>> 9df8fc6 (Initial commit for Illuminy Streamlit app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

<<<<<<< HEAD
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()
=======
# Helper functions (same as before)...

def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents
>>>>>>> 9df8fc6 (Initial commit for Illuminy Streamlit app)

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

<<<<<<< HEAD
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
=======
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

def ask_illuminy_general(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful academic research assistant named Illuminy."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content

def academic_write_and_explain(text):
    prompt = f"""
You are Illuminy, an expert academic writing coach.

Your task is to take the student's writing below and:

1. Rewrite it to improve academic style, clarity, and grammar.
2. Explain the main improvements you made in simple terms.
3. Identify places where in-text citations should be added, especially after definitions, theories, or concepts, and explain why citations are important there.

Here is the student text:
\"\"\"
{text}
\"\"\"

Respond in three parts, clearly labeled:

---  
1. Improved Text:

2. Explanation of Improvements:

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
    flagged = []

    for idx, score in enumerate(similarities):
        if score >= threshold:
            flagged.append((score, chunk_texts[idx]))
    return flagged

def generate_pdf_report(title, sections):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)

    for section_title, content in sections:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, section_title, ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, content)
        pdf.ln(5)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# Streamlit UI

st.title("Illuminy AI - Academic Research Assistant")

# Session state for document & vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "qa" not in st.session_state:
    st.session_state.qa = None

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    documents = load_pdf("temp_uploaded.pdf")
    chunks = split_documents(documents)
    vectorstore = build_vector_store(chunks)
    qa = create_qa_chain(vectorstore)

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = chunks
    st.session_state.qa = qa

    st.success(f"PDF loaded and processed: {uploaded_file.name}")

st.subheader("General Chat with Illuminy")
general_question = st.text_area("Ask any academic question:")
if st.button("Ask General Question"):
    if general_question.strip():
        answer = ask_illuminy_general(general_question)
        st.write(answer)
    else:
        st.warning("Please enter a question.")

st.subheader("Ask questions about uploaded document")
doc_question = st.text_area("Ask about the loaded PDF document:")
if st.button("Ask Document Question"):
    if st.session_state.qa is None:
        st.warning("Please upload and process a PDF first.")
    elif doc_question.strip():
        answer = st.session_state.qa.invoke({"query": doc_question})["result"]
        st.write(answer)
    else:
        st.warning("Please enter a question.")

st.subheader("Academic Writing Assistance")
writing_input = st.text_area("Paste your text to improve academically:")
if st.button("Improve Writing"):
    if writing_input.strip():
        help_text = academic_write_and_explain(writing_input)
        st.text_area("Illuminy's Writing Help", help_text, height=300)
    else:
        st.warning("Please enter some text.")

st.subheader("Plagiarism Checker")
plagiarism_input = st.text_area("Paste text to check plagiarism against uploaded document:")
if st.button("Check Plagiarism"):
    if st.session_state.chunks is None:
        st.warning("Please upload and process a PDF first.")
    elif plagiarism_input.strip():
        matches = check_plagiarism(st.session_state.chunks, plagiarism_input)
        if matches:
            for i, (score, excerpt) in enumerate(matches, 1):
                st.markdown(f"🚨 **Match #{i} – Similarity Score:** {score:.2f}")
                st.write(excerpt[:500] + "...")
        else:
            st.success("No significant similarities found.")
    else:
        st.warning("Please enter some text.")

# Export report feature

st.subheader("Export Report")
if st.session_state.chunks and writing_input.strip():
    if st.button("Generate PDF Report"):
        sections = [
            ("Improved Writing", academic_write_and_explain(writing_input)),
            ("Plagiarism Check", "\n\n".join(
                [f"Match #{i+1} (Score: {score:.2f}):\n{excerpt[:500]}..." for i, (score, excerpt) in enumerate(check_plagiarism(st.session_state.chunks, writing_input))]
            ) or "No significant similarities found.")
        ]
        pdf_path = generate_pdf_report("Illuminy Academic Report", sections)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="illuminy_report.pdf")

    if st.button("Download Text Report"):
        text_report = f"=== Improved Writing ===\n\n{academic_write_and_explain(writing_input)}\n\n=== Plagiarism Check ===\n\n"
        matches = check_plagiarism(st.session_state.chunks, writing_input)
        if matches:
            for i, (score, excerpt) in enumerate(matches, 1):
                text_report += f"Match #{i} (Score: {score:.2f}):\n{excerpt[:500]}...\n\n"
        else:
            text_report += "No significant similarities found.\n"

        st.download_button("Download Text Report", text_report, file_name="illuminy_report.txt")
>>>>>>> 9df8fc6 (Initial commit for Illuminy Streamlit app)
