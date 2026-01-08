Suicide-detection project

Run preprocessing (terminal):

```bash
python src/preprocess.py --input "data.csv" --output "processed.pt" --max_length 128
```

This will create `processed.pt` containing `input_ids`, `attention_mask`, and `labels` tensors.
