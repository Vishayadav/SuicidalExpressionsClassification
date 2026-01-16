import argparse
from pathlib import Path
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch.nn.functional as F
import os

# 🆕 ADDED (SHAP)
import shap
from shap_utils import get_predict_proba, get_shap_explainer

# --------- GLOBAL MODEL (for API / Apify usage) ----------
_MODEL = None
_TOKENIZER = None
_DEVICE = torch.device('cpu')

def load_model(model_dir: Path, tokenizer_name: str):
    if model_dir.exists():
        print(f'Loading model from {model_dir}')
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    else:
        print(f'Model dir {model_dir} not found — loading pretrained `{tokenizer_name}`')
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
        model = DistilBertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=2)
    model.eval()
    return tokenizer, model

def get_model_and_tokenizer(
    model_dir: str = None,
    tokenizer_name: str = 'distilbert-base-uncased'
):
    global _MODEL, _TOKENIZER

    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent / 'model'

    if _MODEL is None or _TOKENIZER is None:
        tokenizer, model = load_model(model_dir, tokenizer_name)
        model.to(_DEVICE)
        _TOKENIZER = tokenizer
        _MODEL = model

    return _TOKENIZER, _MODEL


def predict_texts(texts, tokenizer, model, max_length: int = 128, device='cpu'):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    results = []
    for i in range(len(texts)):
        results.append({
            'text': texts[i],
            'pred': int(preds[i].item()),
            'prob': float(probs[i, preds[i]].item()),
            'probs': probs[i].tolist()
        })
    return results


def predict_text(text: str):
    tokenizer, model = get_model_and_tokenizer()

    results = predict_texts(
        [text],
        tokenizer,
        model,
        device=_DEVICE
    )

    pred = results[0]['pred']
    return 'suicidal' if pred == 1 else 'non-suicidal'


# 🆕 ADDED (SHAP)
def generate_shap_for_text(text, tokenizer, model, output_dir="shap_html"):
    import time
    os.makedirs(output_dir, exist_ok=True)

    predict_proba = get_predict_proba(model, tokenizer)
    explainer = get_shap_explainer(predict_proba, tokenizer)

    shap_values = explainer([text])
    sv = shap_values[0]

    # Tokens are already aligned
    tokens = sv.data

    # Explain suicidal class (class index = 1)
    shap_html = shap.plots.force(
        sv.base_values[1],
        sv.values[:, 1],
        features=tokens,
        matplotlib=False,
        show=False
    )

    # Unique filename per input (safe for interactive mode)
    timestamp = int(time.time() * 1000)
    output_path = os.path.join(output_dir, f"text_shap_{timestamp}.html")

    shap.save_html(output_path, shap_html)

    print(f"✅ SHAP explanation saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-uncased')
    parser.add_argument('--text', type=str, help='Single input text')
    parser.add_argument('--file', type=str, help='Path to a text file')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--max_length', type=int, default=128)

    # 🆕 ADDED (SHAP)
    parser.add_argument('--shap', action='store_true', help='Generate SHAP explanation')

    args = parser.parse_args()

    tokenizer, model = load_model(Path(args.model_dir), args.tokenizer)
    device = torch.device('cpu')
    model.to(device)

    texts = []

    if args.text:
        texts = [args.text]
    elif args.file:
        p = Path(args.file)
        texts = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    elif args.interactive:
        print('Enter text (empty line to quit):')
        while True:
            t = input('> ')
            if not t:
                break
            texts.append(t)
    else:
        parser.print_help()
        return

    results = predict_texts(texts, tokenizer, model, max_length=args.max_length, device=device)

    for i, r in enumerate(results):
        label = 'suicidal' if r['pred'] == 1 else 'non-suicidal'
        print(f"Label: {label}  Prob: {r['prob']:.3f}")
        print(f"Text: {r['text']}\n")

        # 🆕 ADDED (SHAP)
        if args.shap:
            generate_shap_for_text(r['text'], tokenizer, model)

if __name__ == '__main__':
    main()
