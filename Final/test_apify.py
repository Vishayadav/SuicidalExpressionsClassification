from src.data_collection.instagram_comments import fetch_comments
from src.predict import predict_text  # adjust if your function name or file is different

# 1. Fetch comments
url = "https://www.instagram.com/reel/DP4AO6DkuCL/?igsh=MXgzMHFjbWkxeG9sNw=="
comments = fetch_comments(url, limit=10)

# 2. Classify each comment
results = []
for c in comments:
    text = c["text"]
    username = c["username"]
    
    if text:  # ignore empty comments
        label = predict_text(text)  # your model returns 'suicidal' or 'non-suicidal'
    else:
        label = "No text"

    results.append({
        "username": username,
        "comment": text,
        "label": label
    })

# 3. Print results
for r in results:
    print(r)