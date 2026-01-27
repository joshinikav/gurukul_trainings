import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection(name="products")
    print(f"✅ Successfully loaded collection. Total items: {collection.count()}")
    # Example: Peek at the first 5 items
    print(collection.peek(5))
except Exception as e:
    print(f"Error: {e}")

collection = client.get_collection(name="products")

# Fetch stored data
results = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

print("Number of vectors stored:", len(results["ids"]))

# Show first 1 record clearly
print("\n--- SAMPLE RECORD ---")
print("ID:", results["ids"][0])
print("\nDocument (text used for embedding):")
print(results["documents"][0])

print("\nMetadata:")
print(results["metadatas"][0])

print("\nEmbedding vector length:")
print(len(results["embeddings"][0]))
