import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast


def load_and_clean(path: Path):
    df = pd.read_csv(path)
    # Ensure required columns exist
    if 'usertext' not in df.columns or 'label' not in df.columns:
        raise ValueError('CSV must contain columns: usertext, label')
    # Drop rows with empty text
    df['usertext'] = df['usertext'].fillna('')
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['usertext'].str.strip() != '']
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df


def tokenize_and_save(df, tokenizer_name: str, max_length: int, out_path: Path, batch_size: int = 256):
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    texts = df['usertext'].tolist()
    input_ids_list = []
    attention_list = []
    # Tokenize in batches to avoid huge single allocations
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        input_ids_list.append(enc['input_ids'])
        attention_list.append(enc['attention_mask'])

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_list, dim=0)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    torch.save(data, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Preprocess texts and save tokenized tensors')
    parser.add_argument('--input', type=str, default='data.csv')
    parser.add_argument('--output', type=str, default='processed.pt')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-uncased')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    print(f'Loading {inp}...')
    df = load_and_clean(inp)
    print(f'Rows after cleaning: {len(df)}')
    print('Tokenizing... this may download a tokenizer if not present locally')
    path = tokenize_and_save(df, args.tokenizer, args.max_length, out, batch_size=args.batch_size)
    print(f'Saved tokenized dataset to {path}')


if __name__ == '__main__':
    main()
