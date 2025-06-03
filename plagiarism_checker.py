import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1 – Load PDF and create chunks
def load_and_split_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"✅ Loaded and split {len(chunks)} chunks from PDF.")
    return chunks

# Step 2 – Embed chunks and user input
def embed_texts(texts, embedder):
    return embedder.embed_documents(texts)

def check_plagiarism(chunks, input_text):
    embedder = OpenAIEmbeddings()
    
    # Embed chunks and input text
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_vectors = embed_texts(chunk_texts, embedder)
    input_vector = embedder.embed_query(input_text)

    # Compare input vector to each chunk
    similarities = cosine_similarity([input_vector], chunk_vectors)[0]

    # Report high-similarity chunks
    threshold = 0.75
    flagged = []

    for idx, score in enumerate(similarities):
        if score >= threshold:
            flagged.append((score, chunk_texts[idx]))

    print("\n🔍 Plagiarism Detection Report:")
    if flagged:
        for i, (score, text) in enumerate(flagged, 1):
            print(f"\n🚨 Match #{i} – Similarity Score: {score:.2f}")
            print(f"Excerpt from document:\n{text[:500]}...")
    else:
        print("✅ No significant similarities found. You're in the clear!")

# Step 3 – Ask user for text and compare
if __name__ == "__main__":
    pdf_file = "firstone.pdf"  # Replace with your actual file name
    chunks = load_and_split_pdf(pdf_file)

    print("\n🧪 Illuminy Plagiarism Checker Ready.")
    while True:
        user_input = input("\nPaste your paragraph to check (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting plagiarism checker.")
            break
        check_plagiarism(chunks, user_input)