import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import nltk
from nltk.corpus import stopwords

# 🛠 Install stopwords if not already
nltk.download('stopwords')

# 📄 Sample spam and ham messages
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'message': [
        "Hey! Are we still meeting today?",
        "You won a $500 Amazon gift card! Claim now at spam-link.com",
        "Don't forget to bring your notes to class.",
        "Free entry in a contest! Text WIN to 12345 now.",
        "Happy Birthday! Have a great day ahead.",
        "URGENT! Your account will be locked. Click to verify!"
    ]
}

# 🧾 Create DataFrame
df = pd.DataFrame(data)

# 🧹 Clean text function
def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # remove stopwords
    return ' '.join(words)

# 🧼 Apply cleaning
df['cleaned'] = df['message'].apply(clean_text)

# 🔢 Encode labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 🧪 Split data
X = df['cleaned']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 🧠 Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🤖 Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 📈 Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}")

# 🧪 Predict custom messages
test_messages = [
    "Congratulations! You've won a free trip. Click here!",
    "Can you send me the notes from today's class?",
    "Your package has been shipped. Track it here.",
    "Win an iPhone now by entering this contest!"
]

# Clean and transform messages
test_cleaned = [clean_text(msg) for msg in test_messages]
test_vectors = vectorizer.transform(test_cleaned)
predictions = model.predict(test_vectors)

# 🖨️ Print results
for msg, pred in zip(test_messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"\n📩 Message: {msg}\n👉 Prediction: {label}")

