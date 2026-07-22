import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_collection():
    chroma_client = chromadb.PersistentClient(path="chroma_store")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return chroma_client.get_or_create_collection(
        name="documents",
        embedding_function=embed_fn
    )

def answer_question(question: str) -> dict:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=3)
    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": retrieved_chunks
    }

if __name__ == "__main__":
    result = answer_question("What is RAG used for?")
    print("Answer:", result["answer"])
    print("\nSources used:", len(result["sources"]))