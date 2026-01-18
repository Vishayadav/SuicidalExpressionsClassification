"""
Test emoji handling in the model
"""
from src.predict import predict_texts, get_model_and_tokenizer
import torch

# Test data
test_cases = [
    ("We love you NASA 💙🌊", "With emojis"),
    ("We love you NASA", "Without emojis"),
    ("Hello so beautiful 💙💙", "With emojis"),
    ("Hello so beautiful", "Without emojis"),
]

print("="*70)
print("TESTING EMOJI HANDLING IN MODEL")
print("="*70)

tokenizer, model = get_model_and_tokenizer()

for text, description in test_cases:
    results = predict_texts([text], tokenizer, model, device='cpu')
    result = results[0]
    
    label = 'SUICIDAL' if result['pred'] == 1 else 'NON-SUICIDAL'
    confidence = result['prob']
    
    print(f"\n📝 {description}")
    print(f"   Text: {text}")
    print(f"   Prediction: {label} ({confidence*100:.1f}%)")
    print(f"   Probs: [Non-Suicidal: {result['probs'][0]*100:.1f}%, Suicidal: {result['probs'][1]*100:.1f}%]")

print("\n" + "="*70)
print("ISSUE: Text with emojis gets different prediction than without!")
print("SOLUTION: Need to remove emojis before prediction")
print("="*70)
