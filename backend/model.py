import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentModel:
    def __init__(self, model_path: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.id_to_label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def predict(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            logits = self.model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
        return [
            {"label_id": int(p), "label": self.id_to_label[p], "confidence": float(c)}
            for p, c in zip(preds, confidences)
        ]
