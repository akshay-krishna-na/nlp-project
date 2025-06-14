import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np



nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("track-a.csv")
print(df.head())# displays first few lines of dataset


#----------------SVM start----------------

stop_words = set(stopwords.words('english')) #loading english stop words from stopwords repo
lemmatizer = WordNetLemmatizer()

def preprocess_svm(text): #pre processing for svm
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)



df['data_svm'] = df['text'].apply(preprocess_svm)







print(df['data_svm'].head())

# Vectorization for SVM

vectorizer = TfidfVectorizer(max_features=5000)
X_svm = vectorizer.fit_transform(df['data_svm'])
y = df[['anger', 'fear', 'joy', 'sadness', 'surprise']]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_svm, y, test_size=0.3, random_state=42)


# Training the SVM model


svm_model = OneVsRestClassifier(LinearSVC())
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Evaluating the SVM model
print(classification_report(y_test, y_pred, target_names=y.columns))

#testing my own query with svm
print("Enter your query to test the SVM model:")
input_query = input()
test_query_svm = preprocess_svm(input_query)
test_vector_svm = vectorizer.transform([test_query_svm])        
test_prediction_svm = svm_model.predict(test_vector_svm)
print(f"SVM Prediction for '{input_query}': {dict(zip(y.columns, test_prediction_svm[0]))}")

#----------------SVM end----------------


#----------------BERT start----------------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

def preprocess_bert(text, lowercase=True): #pre processing for bert
    if lowercase:
        text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

df['data_bert'] = df['text'].apply(preprocess_bert)

print(df['data_bert'].head())

# Tokenization for BERT
def bert_tokenize(text):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt' 
    )
    return inputs

label_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise']
labels = df[label_columns].values
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['data_bert'], labels, test_size=0.3, random_state=42)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32) 
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    train_labels_tensor
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    test_labels_tensor
)       
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
# Training the BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)          
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluating the BERT model
model.eval()                
predictions = []    
for batch in test_loader:
    input_ids, attention_mask, _ = [b.to(device) for b in batch]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions.append(logits.sigmoid().cpu().numpy())
predictions = np.vstack(predictions)

y_pred_bert = (predictions > 0.5).astype(int)
print(classification_report(test_labels, y_pred_bert, target_names=label_columns))
# Testing my own query with BERT    
print("Enter your query to test the BERT model:")
input_query_bert = input()



test_query_bert = preprocess_bert(input_query_bert)
test_encoding_bert = tokenizer(test_query_bert, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
with torch.no_grad():
    outputs = model(**test_encoding_bert)
    logits = outputs.logits
    test_prediction_bert = (logits.sigmoid().cpu().numpy() > 0.5).astype(int)

print(f"BERT Prediction for '{input_query_bert}': {dict(zip(label_columns, test_prediction_bert[0]))}")




print("Training BERT model...")



