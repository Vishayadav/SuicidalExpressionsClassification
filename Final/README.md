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

Input Formats Supported

Single Text Input (CLI)
python predict.py --text "I feel completely empty and tired of everything" --shap

Interactive Text Input (Multi Line)
python predict.py --interactive --shap

Instagram Post / Comment Analysis
Input
- Instagram post URL
- Number of comments to analyze

This project integrates SHAP (SHapley Additive exPlanations) to improve model transparency.

What SHAP Provides:

- Token-level contribution visualization
- Identification of words increasing or decreasing suicide risk
- HTML-based interactive explanations