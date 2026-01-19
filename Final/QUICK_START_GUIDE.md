# Quick Reference - Using the Fixed App

## ✅ What's Fixed

1. **Optimized for Free Tier** - Only 1 API call instead of 3
2. **Better Error Messages** - Specific solutions for each error type
3. **Token Status Display** - See if API is configured
4. **Intelligent Scrolling** - 5 scrolls (not 10-30)
5. **Reduced Quota Usage** - ~70% less compute units

---

## 📋 Quick Start

### 1. Verify Setup

```bash
cd Final/
python test_token.py
```

✅ Should show: "Authenticated as: visha2.yadav1@gmail.com"

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Use Instagram Feature

1. Go to "📸 Instagram Link" tab
2. Check token status (should show ✅ green check)
3. Paste Instagram post URL
4. Set slider to **10-20 comments** (start low)
5. Click "🔍 Fetch and Analyze"

---

## ⚠️ Important Limits

**Free Tier Monthly Budget**: $5

- Each API call costs compute units
- Large requests use more units
- 100+ comments requests may fail

**Recommendations**:

- Start with 10 comments
- Wait 1-2 minutes between requests
- Check Apify dashboard for usage: https://console.apify.com/account/usage

---

## 🔧 Troubleshooting

### Error: "User was not found or authentication token is not valid"

**Cause**: Token is invalid or expired
**Solution**:

1. Get new token: https://console.apify.com/account/integrations
2. Update .env file
3. Restart app

### Error: "Quota exceeded"

**Cause**: Monthly $5 budget used up
**Solution**:

1. Wait for next month (resets 1st of month)
2. Upgrade plan: https://apify.com/pricing

### Getting fewer comments than expected

**Cause**: Normal on free tier
**Solution**:

- Try different post (may have fewer comments)
- Start with smaller limits (5-10)
- Check if post is public

---

## 📊 Files Changed

| File                                        | Change                                    |
| ------------------------------------------- | ----------------------------------------- |
| `src/data_collection/instagram_comments.py` | Single optimized API call (was 3 retries) |
| `app.py`                                    | Better error messages + token status      |
| `.env`                                      | Already configured with valid token       |

---

## ✨ What Works Now

✅ Fetches 10-20 comments reliably  
✅ Shows exact error messages  
✅ Displays token status  
✅ Optimized for free tier  
✅ Clear troubleshooting guidance

---

## 💡 Tips for Success

1. **Start small**: Request 10 comments first
2. **Check URL**: Paste from browser address bar
3. **Wait between requests**: 2+ minutes between tries
4. **Monitor usage**: Check Apify console
5. **Use public posts**: Private posts won't work

---

**Status**: ✅ Ready to use
**Token**: ✅ Valid for visha2.yadav1@gmail.com
**Mode**: 🎯 Optimized for free tier ($5/month budget)
