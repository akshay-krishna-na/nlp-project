import pandas as pd
import torch
import numpy as np
import re
from transformers import BertTokenizer, BertForSequenceClassification

# Load our saved model and tokenizer for bert
model_path = r"C:\Users\aksha\Documents\NLP project\trained_models\saved_bert_model"  
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# Predict function for one text input
def predict_single(text):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    return (probs > 0.5).astype(int).tolist()

# Main predict function as required
def predict(csv_file_path):
    df = pd.read_csv(csv_file_path)
    predictions = []
    for text in df["text"]:
        pred = predict_single(text)
        predictions.append(pred)
    return np.array(predictions)


if __name__ == "__main__":
    preds = predict("track-a.csv")
    np.set_printoptions(threshold=np.inf)

    print(preds)