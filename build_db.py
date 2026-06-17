from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore

pdf_path = "uploads/Hasini_Kolasani_Resume.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

create_vectorstore(chunks)

print("Database created successfully!")