"""
Debug script to diagnose Instagram comment fetching issues
"""
import sys
from src.data_collection.instagram_comments import fetch_comments
import json

# Test with a sample URL
test_url = input("Enter Instagram post URL: ")
test_limit = int(input("Enter comment limit to fetch: "))

print(f"\n🔍 Testing with URL: {test_url}")
print(f"📊 Requesting: {test_limit} comments\n")

try:
    comments = fetch_comments(test_url, limit=test_limit)
    
    print(f"✅ Total comments returned: {len(comments)}\n")
    print("=" * 80)
    
    # Analyze structure
    for i, comment in enumerate(comments[:5]):  # Show first 5
        print(f"\n📌 Comment {i+1}:")
        print(f"   Keys: {list(comment.keys())}")
        print(f"   Username: {comment.get('username')} (type: {type(comment.get('username'))})")
        print(f"   Text: {comment.get('text')[:80] if comment.get('text') else 'NONE'}...")
        
    # Check for patterns
    usernames = [c.get('username') for c in comments]
    texts = [c.get('text') for c in comments]
    
    print("\n" + "=" * 80)
    print(f"\n📊 Analysis:")
    print(f"   - Total comments: {len(comments)}")
    print(f"   - Username field present: {usernames[0] is not None if comments else 'N/A'}")
    print(f"   - Unique usernames: {len(set(u for u in usernames if u))}")
    print(f"   - None/Empty usernames: {sum(1 for u in usernames if not u)}")
    print(f"   - Comments with text: {sum(1 for t in texts if t)}")
    print(f"   - Comments without text: {sum(1 for t in texts if not t)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
