"""
Comprehensive Apify Debug Script
Debug what the Instagram comment scraper actor is actually returning
"""
import os
import sys
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
    sys.exit(1)

# Get Instagram URL from user
post_url = input("\n📱 Enter Instagram post URL: ").strip()
if not post_url:
    print("❌ URL is required")
    sys.exit(1)

print(f"\n{'='*70}")
print("APIFY INSTAGRAM COMMENT SCRAPER - COMPREHENSIVE DEBUG")
print(f"{'='*70}")
print(f"Testing URL: {post_url}\n")

client = ApifyClient(apify_token)

# Test 1: Minimal parameters (default behavior)
print("\n1️⃣ TEST 1: Minimal Parameters (Just directUrls)")
print("-" * 70)
try:
    run_input = {
        "directUrls": [post_url],
    }
    print(f"Parameters: {json.dumps(run_input, indent=2)}")
    
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    comments = [item for item in items if item.get("text") and item.get("text").strip()]
    
    print(f"✅ Total items returned: {len(items)}")
    print(f"✅ Comments with text: {len(comments)}")
    
    if items:
        print(f"\n📋 First item structure:")
        print(json.dumps(items[0], indent=2, default=str)[:500])
        
except Exception as e:
    print(f"❌ Error: {str(e)[:200]}")

# Test 2: With resultsLimit
print("\n\n2️⃣ TEST 2: With resultsLimit=100")
print("-" * 70)
try:
    run_input = {
        "directUrls": [post_url],
        "resultsLimit": 100,
    }
    print(f"Parameters: {json.dumps(run_input, indent=2)}")
    
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    comments = [item for item in items if item.get("text") and item.get("text").strip()]
    
    print(f"✅ Total items returned: {len(items)}")
    print(f"✅ Comments with text: {len(comments)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)[:200]}")

# Test 3: With maxScrolls
print("\n\n3️⃣ TEST 3: With maxScrolls=10")
print("-" * 70)
try:
    run_input = {
        "directUrls": [post_url],
        "maxScrolls": 10,
    }
    print(f"Parameters: {json.dumps(run_input, indent=2)}")
    
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    comments = [item for item in items if item.get("text") and item.get("text").strip()]
    
    print(f"✅ Total items returned: {len(items)}")
    print(f"✅ Comments with text: {len(comments)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)[:200]}")

# Test 4: Both resultsLimit and maxScrolls
print("\n\n4️⃣ TEST 4: With resultsLimit=200 AND maxScrolls=15")
print("-" * 70)
try:
    run_input = {
        "directUrls": [post_url],
        "resultsLimit": 200,
        "maxScrolls": 15,
    }
    print(f"Parameters: {json.dumps(run_input, indent=2)}")
    
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    comments = [item for item in items if item.get("text") and item.get("text").strip()]
    
    print(f"✅ Total items returned: {len(items)}")
    print(f"✅ Comments with text: {len(comments)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)[:200]}")

# Test 5: Check if there's a scrollWaitSecs parameter
print("\n\n5️⃣ TEST 5: With maxScrolls=15 AND scrollWaitSecs=2000")
print("-" * 70)
try:
    run_input = {
        "directUrls": [post_url],
        "maxScrolls": 15,
        "scrollWaitSecs": 2000,  # Wait for comments to load
    }
    print(f"Parameters: {json.dumps(run_input, indent=2)}")
    
    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    comments = [item for item in items if item.get("text") and item.get("text").strip()]
    
    print(f"✅ Total items returned: {len(items)}")
    print(f"✅ Comments with text: {len(comments)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)[:200]}")

print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("""
Based on these tests, we can determine:
1. The actual parameter names that work
2. The maximum comments that can be scraped from this post
3. Whether scrolling parameters help

Use these results to update instagram_comments.py with the correct parameters.
""")
