import torch
import shap
import numpy as np

# Ensure eval mode
def get_predict_proba(model, tokenizer, device="cpu", max_length=128):
    model.eval()

    def predict_proba(texts):
        texts = [str(t) for t in texts]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    return predict_proba


def get_shap_explainer(predict_proba, tokenizer):
    explainer = shap.Explainer(
        predict_proba,
        masker=shap.maskers.Text(tokenizer)
    )
    return explainer
