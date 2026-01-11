from apify_client import ApifyClient

APIFY_TOKEN = "apify_api_GG4bY9axacxmdveEEsQSYfxcXHe49G0TbwrH"

def fetch_comments(post_url, limit=50):
    client = ApifyClient(APIFY_TOKEN)

    run_input = {
        "directUrls": ["https://www.instagram.com/reel/DP4AO6DkuCL/?igsh=MXgzMHFjbWkxeG9sNw=="],
        "resultsLimit": 10
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
