import pandas as pd

# =========================
# Load & Clean Dataset
# =========================
def clean_amazon_reviews(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Rename columns for consistency
    df = df.rename(columns={
        "Product Name": "product_name",
        "Brand Name": "brand",
        "Price": "price",
        "Rating": "rating",
        "Reviews": "review_text",
        "Review Votes": "helpful_votes"
    })

    # Drop rows where IMPORTANT fields are NULL
    df = df.dropna(subset=[
        "product_name",
        "brand",
        "review_text",
        "rating"
    ])

    # Basic text cleanup
    df["review_text"] = (
        df["review_text"]
        .astype(str)
        .str.replace("\n", " ")
        .str.replace("\r", " ")
        .str.strip()
    )

    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["brand"] = df["brand"].astype(str).str.strip()

    # Optional: reset index after cleaning
    df = df.reset_index(drop=True)

    return df


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    csv_path = r"C:\Users\Joshinika\Downloads\Amazon_Unlocked_Mobile.csv"  # change if needed

    cleaned_df = clean_amazon_reviews(csv_path)
    cleaned_df.to_csv(
        r"C:\Users\Joshinika\Downloads\Amazon_mobile_reviews.csv",
        index=False
    )

    print("✅ Data cleaning completed")
    print("Total rows after cleaning:", len(cleaned_df))
    print("\nSample cleaned record:\n")
    print(cleaned_df.head(1))
