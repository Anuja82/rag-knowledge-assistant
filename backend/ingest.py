from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def store_chunks(chunks):
    client = chromadb.PersistentClient(path="chroma_store")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embed_fn
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection

if __name__ == "__main__":
    raw_text = load_pdf("data/sample.pdf")
    chunks = chunk_text(raw_text)
    print(f"Total chunks created: {len(chunks)}")

    collection = store_chunks(chunks)
    print(f"Stored {collection.count()} chunks in Chroma")

    # quick test search
    results = collection.query(query_texts=["What is RAG used for?"], n_results=2)
    print("\nTest search results:")
    for doc in results["documents"][0]:
        print("-", doc[:150], "...")