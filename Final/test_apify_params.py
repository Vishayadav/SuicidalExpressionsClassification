"""
Direct test of Apify Instagram Comment Scraper
Run this to debug what parameters actually work
"""
import os
from apify_client import ApifyClient
from dotenv import load_dotenv
from pathlib import Path
import json

# Load environment
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

apify_token = os.getenv("APIFY_API_TOKEN")
if not apify_token:
    print("❌ APIFY_API_TOKEN not found in .env")
    exit(1)

# Get Instagram URL from user
post_url = input("\n📱 Enter Instagram post URL: ").strip()
limit = int(input("📊 How many comments to fetch: ").strip() or "26")

print(f"\n🔍 Testing Apify Instagram Comment Scraper")
print(f"   URL: {post_url}")
print(f"   Limit: {limit}")

client = ApifyClient(apify_token)

# Test different parameter combinations
test_configs = [
    {
        "name": "Config 1: resultsLimit only",
        "params": {
            "directUrls": [post_url],
            "resultsLimit": limit,
        }
    },
    {
        "name": "Config 2: commentsLimit + resultsLimit",
        "params": {
            "directUrls": [post_url],
            "commentsLimit": limit,
            "resultsLimit": limit,
        }
    },
    {
        "name": "Config 3: With maxScrolls",
        "params": {
            "directUrls": [post_url],
            "resultsLimit": limit,
            "maxScrolls": 10,
        }
    },
]

for config in test_configs:
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"Parameters: {json.dumps(config['params'], indent=2)}")
    print(f"{'='*60}")
    
    try:
        run = client.actor("apify/instagram-comment-scraper").call(
            run_input=config['params']
        )
        
        print(f"✅ Actor run ID: {run['id']}")
        
        # Count items
        total_items = 0
        comments_with_text = 0
        
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            total_items += 1
            
            # Check if has text
            text = (
                item.get("text") or 
                item.get("comment") or
                item.get("message") or
                item.get("content") or
                ""
            )
            
            if text and isinstance(text, str) and text.strip():
                comments_with_text += 1
                
                # Print first 3 items
                if comments_with_text <= 3:
                    print(f"\n📝 Item {comments_with_text}:")
                    print(f"   Username: {item.get('username')}")
                    print(f"   Text: {text[:80]}...")
        
        print(f"\n📊 Results:")
        print(f"   Total items returned: {total_items}")
        print(f"   Items with text: {comments_with_text}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

print(f"\n{'='*60}")
print("Test complete!")
