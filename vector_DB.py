import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# === Load environment variables ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# === Load enriched JSON file ===
with open("enriched_articles.json", "r", encoding="utf-8") as f:
    docs_json = json.load(f)

# === Convert to LangChain Document objects ===
docs = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in docs_json
]

# === Load sentence-transformer model for embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # lightweight & accurate for legal Q&A
    model_kwargs={'device': 'cpu'},  # set to 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)

# === Set Pinecone index name ===
index_name = "ak-lawbot"

# === Initialize Pinecone Vector Store ===
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

# === Upload documents in batches ===
batch_size = 20  # Suitable for legal text, keeps RAM usage low

for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"✅ Uploaded batch {i // batch_size + 1} of {len(docs) // batch_size + 1}")
