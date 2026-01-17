import os
from apify_client import ApifyClient
from dotenv import load_dotenv
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env from project root (Final/)
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def fetch_comments(post_url: str, limit: int = 50, debug: bool = False):
    """
    Fetch comments from Instagram using Apify actor.
    Uses aggressive scrolling and retries if needed.
    
    Args:
        post_url: Instagram post URL
        limit: Maximum number of comments to fetch
        debug: If True, print debug information
    """
    apify_token = os.getenv("APIFY_API_TOKEN")

    if not apify_token:
        raise RuntimeError(
            "APIFY_API_TOKEN not found. Set it as an environment variable."
        )

    client = ApifyClient(apify_token)

    def run_scraper(max_scrolls: int, results_limit: int):
        """Run the scraper with specified parameters"""
        run_input = {
            "directUrls": [post_url],
            "maxScrolls": max_scrolls,
            "resultsLimit": results_limit,
        }
        
        if debug:
            logger.info(f"🔍 Running Apify with maxScrolls={max_scrolls}, resultsLimit={results_limit}")

        run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
        
        all_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        return all_items

    # Try with increasingly aggressive parameters
    attempts = [
        {"max_scrolls": 10, "results_limit": limit * 2, "desc": "Moderate scrolling"},
        {"max_scrolls": 20, "results_limit": limit * 3, "desc": "Aggressive scrolling"},
        {"max_scrolls": 30, "results_limit": limit * 5, "desc": "Very aggressive scrolling"},
    ]
    
    all_items = []
    
    for attempt in attempts:
        if debug:
            logger.info(f"📊 Attempt: {attempt['desc']}")
        
        try:
            all_items = run_scraper(attempt["max_scrolls"], attempt["results_limit"])
            
            if debug:
                logger.info(f"✅ Got {len(all_items)} items from Apify")
            
            # If we got enough items, stop trying
            valid_comments = [item for item in all_items 
                            if (item.get("text") or item.get("comment") or item.get("message") or item.get("content") or item.get("commentText"))
                            and isinstance((item.get("text") or item.get("comment") or item.get("message") or item.get("content") or item.get("commentText")), str)
                            and (item.get("text") or item.get("comment") or item.get("message") or item.get("content") or item.get("commentText")).strip()]
            
            if len(valid_comments) >= limit:
                if debug:
                    logger.info(f"✅ Got {len(valid_comments)} valid comments - stopping attempts")
                break
            elif debug:
                logger.info(f"⚠️ Only got {len(valid_comments)} valid comments, trying more aggressive approach...")
                
        except Exception as e:
            if debug:
                logger.error(f"❌ Attempt failed: {str(e)[:100]}")
            continue

    comments = []
    collected_count = 0
    skipped_empty = 0
    
    # Extract comments from all items
    for item in all_items:
        # Stop when we've collected enough
        if collected_count >= limit:
            if debug:
                logger.info(f"⏹️ Reached limit of {limit} comments")
            break
            
        # Handle different possible field names for username
        username = (
            item.get("username") or 
            item.get("ownerUsername") or
            item.get("author") or
            item.get("authorName") or
            item.get("ownerName") or
            "Unknown"
        )
        
        # Handle different possible field names for text/comment
        text = (
            item.get("text") or 
            item.get("comment") or
            item.get("message") or
            item.get("content") or
            item.get("commentText") or
            ""
        )
        
        # Only add if there's actual text content
        if text and isinstance(text, str) and text.strip():
            comments.append({
                "username": username,
                "text": text
            })
            collected_count += 1
            
            if debug and collected_count % 5 == 0:
                logger.info(f"📊 Collected {collected_count} comments so far...")
        else:
            skipped_empty += 1

    if debug:
        logger.info(f"📈 Final: Total items from Apify: {len(all_items)}, Valid comments: {len(comments)}, Skipped (no text): {skipped_empty}")
    
    return comments
