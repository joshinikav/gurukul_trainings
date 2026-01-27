import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="products")

# =========================
# Take user input
# =========================
query_text = input("\nAsk your product question: ")

# Convert query to embedding
query_embedding = model.encode(query_text).tolist()

# Semantic search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# Display results
print("\n🔍 Query:", query_text)
print("\nTop Matching Products:\n")

for i in range(len(results["ids"][0])):
    print(f"--- Result {i+1} ---")
    print(results["documents"][0][i])
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])
    print()
