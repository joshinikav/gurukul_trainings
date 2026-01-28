import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# =========================
# CONFIG
# =========================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aigurukul@2.0"

CHROMA_PATH = "./chroma_reviews"
COLLECTION_NAME = "product_reviews_embeddings"

EMBED_MODEL = "all-MiniLM-L6-v2"

# =========================
# CONNECT TO NEO4J
# =========================
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================
# FETCH DATA FROM NEO4J
# =========================
def fetch_reviews():
    query = """
    MATCH (p:Product)-[:HAS_REVIEW]->(r:Review)-[:HAS_BRAND|OF_BRAND*0..1]->(b:Brand)
    RETURN
        p.name AS product,
        b.name AS brand,
        r.text AS review,
        r.rating AS rating
    """
    with driver.session() as session:
        return [row.data() for row in session.run(query)]

# =========================
# BUILD EMBEDDING TEXT
# =========================
def build_embedding_text(row):
    return f"""
Product: {row['product']}
Brand: {row['brand']}
Rating: {row['rating']}
Review: {row['review']}
""".strip()

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("🔹 Loading data from Neo4j...")
    rows = fetch_reviews()
    print(f"🔹 Total reviews fetched: {len(rows)}")

    # Load embedding model
    model = SentenceTransformer(EMBED_MODEL)

    # Init Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for i, row in enumerate(rows):
        text = build_embedding_text(row)
        vector = model.encode(text).tolist()

        documents.append(text)
        embeddings.append(vector)
        ids.append(f"review_{i}")

        metadatas.append({
            "product": row["product"],
            "brand": row["brand"],
            "rating": int(row["rating"])
        })

    # Store ALL embeddings (NO LIMIT)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Stored {collection.count()} embeddings in ChromaDB")

    # =========================
    # PROOF: VIEW ONE VECTOR
    # =========================
    sample = collection.get(
        ids=[ids[0]],
        include=["documents", "metadatas", "embeddings"]
    )

    print("\n📌 SAMPLE VECTOR PROOF")
    print("\nDocument:\n", sample["documents"][0])
    print("\nMetadata:", sample["metadatas"][0])
    print("Embedding length:", len(sample["embeddings"][0]))

    # =========================
    # SIMPLE SEMANTIC SEARCH
    # =========================
    while True:
        query = input("\nAsk your question (type exit to quit): ").strip()
        if query.lower() == "exit":
            break

        q_vector = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[q_vector],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        print("\n🔍 Semantic Results\n")
        for i in range(len(results["documents"][0])):
            print(f"Result {i+1}")
            print(results["documents"][0][i])
            print("Metadata:", results["metadatas"][0][i])
            print("Distance:", results["distances"][0][i])
            print("-" * 40)

    driver.close()
