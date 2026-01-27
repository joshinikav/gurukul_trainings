from neo4j import GraphDatabase

# =========================
# Neo4j connection details
# =========================
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aigurukul@2.0"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================
# Fetch products from Neo4j
# =========================
def fetch_products(limit=5):
    query = """
    MATCH (p:Product)
    RETURN 
        p.`Product Name` AS product_name,
        p.`Brand Name` AS brand,
        p.Price AS price,
        p.Rating AS rating
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, limit=limit)
        return [record.data() for record in result]

# =========================
# Converts structured data to semantic embedding text
# =========================
def build_embedding_text(product):
    price = product.get("price")
    rating = product.get("rating")

    # Price category
    if price is None:
        price_category = "unknown price range"
    elif price < 100:
        price_category = "budget-friendly"
    elif price < 300:
        price_category = "mid-range"
    else:
        price_category = "premium"

    # Rating sentiment
    if rating is None:
        rating_sentiment = "no customer rating available"
    elif rating <= 2:
        rating_sentiment = "poorly rated"
    elif rating == 3:
        rating_sentiment = "moderately rated"
    else:
        rating_sentiment = "highly rated"

    return f"""
This product is a mobile phone.

Product name: {product.get('product_name')}.
Brand: {product.get('brand') or 'Unknown'}.

It is priced at {price}, making it a {price_category} device.
The product is {rating_sentiment} with a customer rating of {rating} out of 5.
""".strip()

# =========================
# Main execution
# =========================
if __name__ == "__main__":
    products = fetch_products(limit=5)

    print("\n========== RAW PRODUCT DATA ==========")
    for i, p in enumerate(products, 1):
        print(f"\nProduct {i}")
        for k, v in p.items():
            print(f"{k}: {v}")

    print("\n========== EMBEDDING TEXT ==========")
    for i, p in enumerate(products, 1):
        embedding_text = build_embedding_text(p)
        print(f"\nEmbedding Text {i}:\n{embedding_text}\n")

    driver.close()
