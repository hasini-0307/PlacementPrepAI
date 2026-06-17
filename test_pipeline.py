from src.loader import load_pdf
from src.splitter import split_documents

pdf_path = "uploads/Hasini_Kolasani_Resume.pdf" 

documents = load_pdf(pdf_path)

print(f"Number of pages: {len(documents)}")

chunks = split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

print("\nMetadata:\n")
print(chunks[0].metadata)