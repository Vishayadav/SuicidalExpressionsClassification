"""
Test Apify Token Validity
"""
import os
from apify_client import ApifyClient
from dotenv import load_dotenv
from pathlib import Path

# Load environment
ENV_PATH = Path(__file__).resolve().parent / ".env"
print(f"Looking for .env at: {ENV_PATH}")
print(f"File exists: {ENV_PATH.exists()}")

load_dotenv(dotenv_path=ENV_PATH)

token = os.getenv("APIFY_API_TOKEN")
print(f"\n🔑 Token loaded: {token[:20]}...{token[-10:] if token else 'NONE'}")
print(f"Token length: {len(token) if token else 0}")

if not token:
    print("❌ ERROR: APIFY_API_TOKEN not found in .env file!")
    exit(1)

try:
    print("\n🔍 Testing Apify API connection...")
    client = ApifyClient(token)
    
    # Try to get user info
    user = client.user().get()
    print(f"✅ SUCCESS! Authenticated as: {user.get('email', 'Unknown')}")
    print(f"📊 Account status: {user.get('plan', 'Unknown')}")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("\n🔧 Solutions:")
    print("1. Token is invalid or expired - get a new one from https://console.apify.com")
    print("2. Token has wrong format - check if it starts with 'apify_api_'")
    print("3. Network issue - check your internet connection")
