import os
from apify_client import ApifyClient
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (Final/)
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def fetch_comments(post_url: str, limit: int = 10):
    apify_token = os.getenv("APIFY_API_TOKEN")

    if not apify_token:
        raise RuntimeError(
            "APIFY_API_TOKEN not found. Set it as an environment variable."
        )

    client = ApifyClient(apify_token)

    run_input = {
        "directUrls": [post_url],
        "resultsLimit": limit
    }

    run = client.actor("apify/instagram-comment-scraper").call(
        run_input=run_input
    )

    comments = []

    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        comments.append({
            "username": item.get("username"),
            "text": item.get("text")
        })

    return comments
