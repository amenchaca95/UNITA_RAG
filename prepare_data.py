# prepare_data.py

import os
from pathlib import Path
# import fitz  # PyMuPDF — no longer needed if we skip PDF conversion
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
txt_files = [
    "UNITA1.txt",
    "UNITA2.txt",
    "UNITA3.txt",
    "UNITA4.txt",
    "UNITA5.txt"
]
persist_path = "./chroma_store"

# --- PDF to TXT conversion (now commented out, done manually) ---
# pdf_path = "Université — Collectives - UniTA Project Datacloud.pdf"
# txt_path = "UNITA.txt"
# if not Path(txt_path).exists():
#     print("📄 Converting PDF to plain text...")
#     doc = fitz.open(pdf_path)
#     with open(txt_path, "w", encoding="utf-8") as f:
#         for page in doc:
#             f.write(page.get_text("text") + "\n\n")
#     print(f"✅ Saved as {txt_path}")

# --- Load and merge all documents ---
documents = []
for txt_file in txt_files:
    if Path(txt_file).exists():
        print(f"📥 Loading {txt_file}")
        loader = TextLoader(txt_file, encoding="utf-8")
        documents.extend(loader.load())
    else:
        print(f"⚠️ File not found: {txt_file}")

# --- Split documents ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents)

# --- Generate and persist embeddings ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_path)
vectorstore.persist()

print("✅ Embeddings created and saved.")
