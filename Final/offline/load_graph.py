#load_graph.py
import pandas as pd
import os
from neo4j import GraphDatabase

# ---------------- PATH CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_PATH = os.path.join(
    BASE_DIR, "data", "cleaned", "amazon_reviews_cleaned.csv"
)

# ---------------- NEO4J CONFIG ----------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aigurukul@2.0" 

driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ---------------- CYPHER WRITE FUNCTION ----------------
def create_graph(tx, brand, product, price, review_id, rating, votes):
    tx.run("""
        MERGE (b:Brand {name: $brand})
        MERGE (p:Product {name: $product})
        SET p.price = $price
        MERGE (b)-[:MAKES]->(p)
        CREATE (r:Review {
            review_id: $review_id,
            rating: $rating,
            votes: $votes
        })
        MERGE (p)-[:HAS_REVIEW]->(r)
    """, {
        "brand": brand,
        "product": product,
        "price": price,
        "review_id": review_id,
        "rating": rating,
        "votes": votes
    })

# ---------------- LOAD DATA ----------------
def load_graph():
    print("📥 Loading cleaned data...")
    df = pd.read_csv(CLEAN_PATH)

    print(f"📊 Rows to insert: {len(df)}")

    with driver.session() as session:
        for idx, row in df.iterrows():
            review_id = f"rev_{idx:08d}"

            session.execute_write(
                create_graph,
                row["brand"],
                row["product"],
                row.get("price"),
                review_id,
                row["rating"],
                row["votes"]
            )

    print("✅ Graph loaded successfully")

if __name__ == "__main__":
    load_graph()
