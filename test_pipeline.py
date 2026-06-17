from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore

pdf_path = "uploads/Hasini_Kolasani_Resume.pdf"

# Load PDF
documents = load_pdf(pdf_path)

print(f"Pages: {len(documents)}")

# Split documents
chunks = split_documents(documents)

print(f"Chunks: {len(chunks)}")

# Create vector database
vectorstore = create_vectorstore(chunks)

print("\nVector database created successfully!")