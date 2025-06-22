from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
model_path=r"C:\Users\akshay\OneDrive\Desktop\nlp-project\trained_models\saved_bert_model" #replace accordingly with model directory
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preprocess and predict
def preprocess_bert(text):
    import re
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def predict_bert(text):
    text = preprocess_bert(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    return dict(zip(['anger', 'fear', 'joy', 'sadness', 'surprise'], (probs > 0.5).astype(int)))

# Example
text = input("Enter you text for analysis :")


print("Analysis result with BERT model : ")
print(predict_bert(text))