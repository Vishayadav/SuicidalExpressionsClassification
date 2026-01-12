import sys
from src.data_collection.instagram_comments import fetch_comments
from src.predict import predict_text


def main():
    # -------------------------------
    # 1. Read arguments
    # -------------------------------
    if len(sys.argv) < 2:
        print("Usage: python test_apify.py <instagram_post_url> [limit]")
        sys.exit(1)

    url = sys.argv[1]

    # limit is optional
    if len(sys.argv) >= 3:
        try:
            limit = int(sys.argv[2])
        except ValueError:
            print("Limit must be an integer")
            sys.exit(1)
    else:
        limit = 10  # default

    # -------------------------------
    # 2. Fetch comments
    # -------------------------------
    print(f"\nFetching {limit} comments from:\n{url}\n")
    comments = fetch_comments(url, limit=limit)

    # -------------------------------
    # 3. Classify comments
    # -------------------------------
    results = []
    for c in comments:
        text = c.get("text", "")
        username = c.get("username")

        if text.strip():
            label = predict_text(text)
        else:
            label = "No text"

        results.append({
            "username": username,
            "comment": text,
            "label": label
        })

    # -------------------------------
    # 4. Output
    # -------------------------------
    print("\n===== CLASSIFICATION RESULTS =====\n")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
