#clean_data.py
import pandas as pd
import os

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(
    BASE_DIR, "data", "raw", "Amazon_Unlocked_Mobile.csv"
)

CLEAN_PATH = os.path.join(
    BASE_DIR, "data", "cleaned", "amazon_reviews_cleaned.csv"
)

def clean_data():
    print("📥 Loading raw dataset...")
    df = pd.read_csv(RAW_PATH, low_memory=False)

    print(f"📊 Initial row count: {len(df)}")

    df = df.rename(columns={
        "Product Name": "product",
        "Brand Name": "brand",
        "Price": "price",
        "Rating": "rating",
        "Reviews": "review_text",
        "Review Votes": "votes"
    })

    mandatory_columns = ["product", "brand", "review_text", "rating"]
    df = df.dropna(subset=mandatory_columns)

    print(f"🧹 After dropping mandatory fields: {len(df)}")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df[df["rating"].between(1, 5)]

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["review_text"] = df["review_text"].str.slice(0, 1000)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(CLEAN_PATH, index=False)
    print("✅ Cleaned data saved to:", CLEAN_PATH)

if __name__ == "__main__":
    clean_data()
