import joblib

# Load model and vectorizer

svm_model = joblib.load(r"C:\Users\akshay\OneDrive\Desktop\nlp-project\trained_models\saved_svm_model\svm_model.pkl")#replace accordingly with model directory
vectorizer = joblib.load(r"C:\Users\akshay\OneDrive\Desktop\nlp-project\trained_models\saved_svm_model\svm_vectorizer.pkl")#replace accordingly with model directory

# Preprocess input (use the same function you defined)
def preprocess_svm(text):
    import re, string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Predict
text = input("Enter you text for analysis :")
processed = preprocess_svm(text)
vector = vectorizer.transform([processed])
prediction = svm_model.predict(vector)

print("Analysis result with BERT model : ")
print(dict(zip(['anger', 'fear', 'joy', 'sadness', 'surprise'], prediction[0])))