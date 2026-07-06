from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading PDF...")

loader = PyPDFLoader("law.pdf")
documents = loader.load()

print("Splitting document...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# Add metadata BEFORE creating FAISS
for doc in docs:
    doc.metadata["source"] = "IPC"

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating FAISS vector database...")

db = FAISS.from_documents(docs, embeddings)

db.save_local("vector_db")