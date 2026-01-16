import sys
import os
import torch
import shap
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 🆕 ADDED: Instagram comment fetcher
from data_collection.instagram_comments import fetch_comments

# -------------------------------
# 1️⃣ Read terminal arguments
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python test_shap.py <instagram_post_url> [comment_limit]")
    sys.exit(1)

INSTAGRAM_URL = sys.argv[1]
COMMENT_LIMIT = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

# -------------------------------
# 2️⃣ Load model and tokenizer
# -------------------------------
MODEL_PATH = "model"  # Update if needed
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("✅ Model and tokenizer loaded")

# -------------------------------
# 3️⃣ Fetch Instagram comments
# -------------------------------
comments = fetch_comments(INSTAGRAM_URL, COMMENT_LIMIT)
texts = [c["text"] for c in comments if c.get("text")]

if not texts:
    raise RuntimeError("No valid comments fetched from Instagram")

print(f"✅ Fetched {len(texts)} comments")

# -------------------------------
# 4️⃣ Prediction function for raw text
# -------------------------------
def predict_fn(texts_list):
    # Ensure input is list of strings
    # Convert NumPy arrays to strings if needed
    texts_list = [str(x) if not isinstance(x, str) else x for x in texts_list]

    inputs = tokenizer(
        texts_list,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()

print("✅ Prediction function ready")

# -------------------------------
# 5️⃣ Create SHAP explainer
# -------------------------------
explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text(tokenizer))
print("✅ SHAP explainer created")

# -------------------------------
# 6️⃣ Compute SHAP values
# -------------------------------
shap_values = explainer(texts)
print("✅ SHAP values computed")

# -------------------------------
# 7️⃣ Save SHAP HTML plots
# -------------------------------
OUTPUT_DIR = "shap_html"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, sv in enumerate(shap_values):
    html = shap.plots.text(sv, display=False)

    path = os.path.join(OUTPUT_DIR, f"shap_plot_{i}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Saved SHAP text plot: {path}")


# -------------------------------
# 8️⃣ Save predictions CSV
# -------------------------------
preds = predict_fn(texts)
labels = preds.argmax(axis=1)

df = pd.DataFrame({
    "username": [c["username"] for c in comments],
    "comment": texts,
    "label": ["suicidal" if l == 1 else "non-suicidal" for l in labels],
    "shap_file": [f"shap_plot_{i}.html" for i in range(len(texts))]
})

df.to_csv("instagram_predictions.csv", index=False)
print("✅ Saved instagram_predictions.csv")
print("🎉 Instagram → Model → SHAP pipeline completed")
