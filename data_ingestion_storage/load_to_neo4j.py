import pandas as pd
import uuid
from neo4j import GraphDatabase

# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\Joshinika\Downloads\Amazon_mobile_reviews.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aigurukul@2.0"

# =========================
# CONNECT
# =========================
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH, low_memory=False)

def create_graph(tx, product, brand, review_text, rating, votes):
    tx.run("""
        MERGE (p:Product {name: $product})
        MERGE (b:Brand {name: $brand})
        MERGE (p)-[:OF_BRAND]->(b)

        CREATE (r:Review {
            id: $review_id,
            text: $review_text,
            rating: $rating,
            votes: $votes
        })

        MERGE (p)-[:HAS_REVIEW]->(r)
    """,
    product=product,
    brand=brand,
    review_text=review_text,
    rating=rating,
    votes=votes,
    review_id=str(uuid.uuid4())
    )

# =========================
# WRITE TO NEO4J
# =========================
with driver.session() as session:
    for _, row in df.iterrows():
        session.execute_write(
            create_graph,
            row["product_name"],
            row["brand"],
            row["review_text"],
            int(row["rating"]),
            int(row["helpful_votes"]) if not pd.isna(row["helpful_votes"]) else 0
        )

print("✅ Knowledge Graph created successfully")

driver.close()
