import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1 – Load the PDF
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {file_path}")
    return documents

# Step 2 – Split into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# Step 3 – Embed & Store in FAISS
def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created")
    return vectorstore

# Step 4 – Ask questions from the document
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Example usage
if __name__ == "__main__":
    file_path = "firstone.pdf"  # Replace with your PDF file name
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    vectorstore = build_vector_store(chunks)
    qa = create_qa_chain(vectorstore)

    print("\n🧠 Illuminy is ready to answer questions based on the PDF!")
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() in ["exit", "quit"]:
            break
        output = qa.invoke({"query": question})
        print("\nIlluminy says:\n", output["result"])