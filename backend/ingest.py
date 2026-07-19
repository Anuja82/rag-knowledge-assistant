from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

if __name__ == "__main__":
    raw_text = load_pdf("data/sample.pdf")
    chunks = chunk_text(raw_text)
    print(f"Total chunks created: {len(chunks)}")
    print("First chunk preview:")
    print(chunks[0])