"""
Test that emoji removal fixes the classification issue
"""
from src.predict import predict_texts, get_model_and_tokenizer
from src.preprocess_text import clean_text
import torch

# Test data - the problematic examples from user
test_cases = [
    ("We love you NASA 💙🌊", "Should now be Non-Suicidal"),
    ("We love you NASA", "Should be Non-Suicidal"),
    ("Hello so beautiful 💙💙", "Should now be Non-Suicidal"),
    ("Hello so beautiful", "Should be Non-Suicidal"),
]

print("="*80)
print("TESTING EMOJI FIX FOR MODEL PREDICTIONS")
print("="*80)

# Show preprocessing
print("\n1️⃣ TEXT PREPROCESSING (Emoji Removal)")
print("-"*80)
for text, _ in test_cases:
    cleaned = clean_text(text)
    print(f"\nOriginal: {text}")
    print(f"Cleaned:  {cleaned}")
    print(f"Changed:  {text != cleaned}")

# Test predictions
print("\n\n2️⃣ MODEL PREDICTIONS (After Fix)")
print("-"*80)

tokenizer, model = get_model_and_tokenizer()
device = 'cpu'

for text, description in test_cases:
    results = predict_texts([text], tokenizer, model, device=device)
    result = results[0]
    
    label = '✅ NON-SUICIDAL' if result['pred'] == 0 else '❌ SUICIDAL'
    confidence = result['prob']
    
    print(f"\n📝 {description}")
    print(f"   Original: {text}")
    print(f"   Cleaned:  {result['cleaned_text']}")
    print(f"   Prediction: {label} ({confidence*100:.1f}%)")

print("\n" + "="*80)
print("✅ EXPECTED RESULT: All should be NON-SUICIDAL")
print("="*80)
