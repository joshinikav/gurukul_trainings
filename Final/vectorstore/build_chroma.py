#build_chroma.py
import os
import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned", "amazon_reviews_cleaned.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "vectorstore", "chroma_amazon_reviews")

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

df = pd.read_csv(CLEAN_PATH).head(10000)

documents = []

for idx, row in df.iterrows():
    review_id = f"rev_{idx:08d}"

    documents.append(
        Document(
            page_content=row["review_text"],
            metadata={
                "review_id": review_id,
                "product": row["product"],
                "brand": row["brand"]
            }
        )
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

embedding = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding,
    persist_directory=CHROMA_DIR,
    collection_name="amazon_reviews"
)

print("Chroma embeddings created successfully")
