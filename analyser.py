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

print("Training BERT model...")

