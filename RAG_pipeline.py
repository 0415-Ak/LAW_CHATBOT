import os
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings  # updated import

# === Load credentials ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# === Build Retrieval Chain Function ===
def get_retrieval_chain():
    # 1. Embedding Model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Vector Store (Pinecone)
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_model,
        pinecone_api_key=PINECONE_API_KEY
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",  # or "mmr" for diversity
        search_kwargs={"k": 20}
    )

    # 3. Mistral LLM
    model = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY)

    # 4. Prompt Template (custom for Indian Constitution)
    prompt = ChatPromptTemplate.from_template("""
You are an expert AI legal assistant that answers questions **only about the Indian Constitution**.

INSTRUCTIONS:
- Use only the provided <context> to answer the question.
- Do NOT use external knowledge.
- If the answer is not found in the context, say:
  "I don't know based on the provided articles of the Indian Constitution."
- Provide short, accurate, legal-style answers.

<context>
{context}
</context>

Question: {input}
Answer:
""")

    # 5. Build the Chain
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("done")

    return retrieval_chain

if __name__ == "__main__":
    chain = get_retrieval_chain()
    print("📜 Indian Constitution Chatbot is running.")
    print("💬 Type your question below (or type 'exit' to quit):\n")

    while True:
        query = input("You: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting the chatbot. Jai Hind 🇮🇳")
            break

        try:
            result = chain.invoke({"input": query})
            print("\n✅ Answer:\n", result["answer"])
        except Exception as e:
            print("⚠️ Error occurred:", str(e))

