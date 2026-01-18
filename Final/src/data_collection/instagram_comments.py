import os
from apify_client import ApifyClient
from dotenv import load_dotenv
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env from project root (Final/)
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def fetch_comments(post_url: str, limit: int = 50, debug: bool = False):
    """
    Fetch comments from Instagram using Apify actor.
    Optimized for free tier with single attempt strategy.
    
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

    # FREE TIER OPTIMIZATION:
    # - Only make ONE API call to avoid quota exhaustion
    # - Use intelligent parameters that work best for free tier
    # - Request more items than needed but make only 1 call
    
    if debug:
        logger.info(f"🔍 Fetching {limit} comments (optimized for free tier)")
        logger.info(f"   Target URL: {post_url}")

    try:
        # Single optimized attempt
        run_input = {
            "directUrls": [post_url],
            "maxScrolls": 5,  # Moderate scrolling (free tier friendly)
            "resultsLimit": limit * 2,  # Request 2x what we need
        }
        
        if debug:
            logger.info(f"📊 Parameters: maxScrolls=5, resultsLimit={limit*2}")

        run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
        
        if debug:
            logger.info(f"✅ Apify run completed: {run['id']}")

        all_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        if debug:
            logger.info(f"📈 Total items from Apify: {len(all_items)}")

    except Exception as e:
        error_msg = str(e)
        if debug:
            logger.error(f"❌ Apify error: {error_msg[:200]}")
        
        # Check for specific error types
        if "authentication" in error_msg.lower() or "token" in error_msg.lower():
            raise RuntimeError(
                f"Authentication failed: Your APIFY_API_TOKEN appears to be invalid or expired.\n"
                f"Error: {error_msg}\n"
                f"Solution: Get a new token from https://console.apify.com/account/integrations"
            )
        elif "quota" in error_msg.lower() or "credits" in error_msg.lower():
            raise RuntimeError(
                f"Quota exceeded: Your free tier monthly credits have been exceeded.\n"
                f"Error: {error_msg}\n"
                f"Solution: Wait for next month or upgrade your Apify plan"
            )
        else:
            raise RuntimeError(f"Apify error: {error_msg}")

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
        logger.info(f"📈 Final: Total items: {len(all_items)}, Valid comments: {len(comments)}, Skipped: {skipped_empty}")
    
    return comments
