import argparse
from pathlib import Path
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch.nn.functional as F


def load_model(model_dir: Path, tokenizer_name: str):
    # If model_dir exists and contains pretrained model, load from there
    if model_dir.exists():
        print(f'Loading model from {model_dir}')
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    else:
        print(f'Model dir {model_dir} not found — loading pretrained `{tokenizer_name}` (not fine-tuned)')
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
        model = DistilBertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=2)
    model.eval()
    return tokenizer, model


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
        results.append({'text': texts[i], 'pred': int(preds[i].item()), 'prob': float(probs[i, preds[i]].item()), 'probs': probs[i].tolist()})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-uncased')
    parser.add_argument('--text', type=str, help='Single input text')
    parser.add_argument('--file', type=str, help='Path to a text file with one example per line')
    parser.add_argument('--interactive', action='store_true', help='Run interactive prompt')
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    tokenizer, model = load_model(Path(args.model_dir), args.tokenizer)
    device = torch.device('cpu')
    model.to(device)

    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            print('File not found:', p)
            return
        texts = [l.strip() for l in p.read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    elif args.interactive:
        print('Enter text (empty line to quit):')
        while True:
            try:
                t = input('> ')
            except EOFError:
                break
            if not t:
                break
            texts.append(t)
    else:
        parser.print_help()
        return

    if not texts:
        print('No texts to predict')
        return

    results = predict_texts(texts, tokenizer, model, max_length=args.max_length, device=device)
    for r in results:
        label = r['pred']
        prob = r['prob']
        print(f"Label: {label}  Prob: {prob:.3f}  Text: {r['text']}")


if __name__ == '__main__':
    main()
