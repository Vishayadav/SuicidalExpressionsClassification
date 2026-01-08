Suicide-detection project

Run preprocessing (terminal):

```bash
python src/preprocess.py --input "data.csv" --output "processed.pt" --max_length 128
```

This will create `processed.pt` containing `input_ids`, `attention_mask`, and `labels` tensors.

---

python src/predict.py --model_dir model_test --text "I feel hopeless and want to end my life"

python src/predict.py --model_dir model_test --interactive

python src/predict.py --model_dir model_test --file samples.txt

Troubleshooting

If you get FileNotFoundError: processed.pt when training, run preprocessing first:
python src/preprocess.py --input "data.csv" --output "processed.pt" --max_length 128 --batch_size 64
