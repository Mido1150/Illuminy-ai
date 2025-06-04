import os
from dotenv import load_dotenv
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --------- PDF Loading and Vector Store ---------
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {file_path}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created")
    return vectorstore

# --------- General Chat ---------
def ask_illuminy_general(prompt):
    try:
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
    except Exception as e:
        return f"Error: {e}"

# --------- Document Q&A ---------
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain

# --------- Writing Assistant ---------
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
    try:
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
    except Exception as e:
        return f"Error: {e}"

# --------- Plagiarism Checker ---------
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

    if flagged:
        print("\n🔍 Plagiarism Detection Report:")
        for i, (score, text) in enumerate(flagged, 1):
            print(f"\n🚨 Match #{i} – Similarity Score: {score:.2f}")
            print(f"Excerpt from document:\n{text[:500]}...")
    else:
        print("\n✅ No significant similarities found. You're in the clear!")

# --------- Main program ---------
def main():
    print("Welcome to Illuminy AI - integrated assistant.")
    vectorstore = None
    qa = None
    chunks = None

    while True:
        print("\nChoose an option:")
        print("1. Load a PDF document")
        print("2. Ask general questions")
        print("3. Ask questions about loaded document")
        print("4. Writing assistant (improve your text)")
        print("5. Check plagiarism against loaded document")
        print("6. Exit")

        choice = input("Your choice: ")

        if choice == "1":
            path = input("Enter PDF file path: ")
            try:
                docs = load_pdf(path)
                chunks = split_documents(docs)
                vectorstore = build_vector_store(chunks)
                qa = create_qa_chain(vectorstore)
            except Exception as e:
                print(f"Error loading PDF: {e}")

        elif choice == "2":
            question = input("Ask Illuminy general question: ")
            answer = ask_illuminy_general(question)
            print("\nIlluminy says:\n", answer)

        elif choice == "3":
            if not qa:
                print("Please load a PDF first.")
                continue
            question = input("Ask about the loaded document: ")
            try:
                answer = qa.invoke({"query": question})["result"]
                print("\nIlluminy says:\n", answer)
            except Exception as e:
                print(f"Error querying document: {e}")

        elif choice == "4":
            text = input("Paste your text for academic writing help: ")
            response = academic_write_and_explain(text)
            print("\nIlluminy's writing help:\n", response)

        elif choice == "5":
            if not chunks:
                print("Please load a PDF first.")
                continue
            text = input("Paste your text to check plagiarism: ")
            check_plagiarism(chunks, text)

        elif choice == "6":
            print("Goodbye! Keep learning and writing.")
            break

        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --------- PDF Loading and Vector Store ---------
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {file_path}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created")
    return vectorstore

# --------- General Chat ---------
def ask_illuminy_general(prompt):
    try:
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
    except Exception as e:
        return f"Error: {e}"

# --------- Document Q&A ---------
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain

# --------- Writing Assistant ---------
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
    try:
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
    except Exception as e:
        return f"Error: {e}"

# --------- Plagiarism Checker ---------
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

    if flagged:
        print("\n🔍 Plagiarism Detection Report:")
        for i, (score, text) in enumerate(flagged, 1):
            print(f"\n🚨 Match #{i} – Similarity Score: {score:.2f}")
            print(f"Excerpt from document:\n{text[:500]}...")
    else:
        print("\n✅ No significant similarities found. You're in the clear!")

# --------- Main program ---------
def main():
    print("Welcome to Illuminy AI - integrated assistant.")
    vectorstore = None
    qa = None
    chunks = None

    while True:
        print("\nChoose an option:")
        print("1. Load a PDF document")
        print("2. Ask general questions")
        print("3. Ask questions about loaded document")
        print("4. Writing assistant (improve your text)")
        print("5. Check plagiarism against loaded document")
        print("6. Exit")

        choice = input("Your choice: ")

        if choice == "1":
            path = input("Enter PDF file path: ")
            try:
                docs = load_pdf(path)
                chunks = split_documents(docs)
                vectorstore = build_vector_store(chunks)
                qa = create_qa_chain(vectorstore)
            except Exception as e:
                print(f"Error loading PDF: {e}")

        elif choice == "2":
            question = input("Ask Illuminy general question: ")
            answer = ask_illuminy_general(question)
            print("\nIlluminy says:\n", answer)

        elif choice == "3":
            if not qa:
                print("Please load a PDF first.")
                continue
            question = input("Ask about the loaded document: ")
            try:
                answer = qa.invoke({"query": question})["result"]
                print("\nIlluminy says:\n", answer)
            except Exception as e:
                print(f"Error querying document: {e}")

        elif choice == "4":
            text = input("Paste your text for academic writing help: ")
            response = academic_write_and_explain(text)
            print("\nIlluminy's writing help:\n", response)

        elif choice == "5":
            if not chunks:
                print("Please load a PDF first.")
                continue
            text = input("Paste your text to check plagiarism: ")
            check_plagiarism(chunks, text)

        elif choice == "6":
            print("Goodbye! Keep learning and writing.")
            break

        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()
    import streamlit as st

# ... your existing imports and code ...

def load_pdf(file):
    # Your existing PDF loading logic, e.g. PyMuPDFLoader
    loader = PyMuPDFLoader(file)
    documents = loader.load()
    return documents

def main():
    st.title("Illuminy Academic Assistant - Multi PDF Support")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_chunks = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily to disk
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and chunk
            chunks = load_pdf("temp_uploaded.pdf")
            all_chunks.extend(chunks)

        st.success(f"Loaded {len(all_chunks)} chunks from {len(uploaded_files)} PDFs.")

        # Build vector store with all chunks combined
        vectorstore = build_vector_store(all_chunks)

        # Continue with your chat interface, querying etc.
        # For example:
        query = st.text_input("Ask Illuminy a question:")
        if query:
            answer = ask_illuminy_with_vectorstore(query, vectorstore)
            st.write("### Illuminy says:")
            st.write(answer)

if __name__ == "__main__":
    main()