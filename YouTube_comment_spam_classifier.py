from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = remove_eng_lang_contraction(text)
    text = re.sub(r'\\r|\\n|\\"', ' ', text)  # Remove escape characters
    #text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stop words
    return text

# Remove contractions
def remove_eng_lang_contraction(text):
    contractions = {
        r"won't": "will not",
        r"can't": "can not",
        r"n't": " not",
        r"'re": " are",
        r"'s": " is",
        r"'d": " would",
        r"'ll": " will",
        r"'t": " not",
        r"'ve": " have",
        r"'m": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return text

# Load the dataset
file_path = Path(__file__).parent / 'data/Youtube01-Psy.csv'
df = pd.read_csv(file_path)

# Check and clean the data
df['CONTENT'] = df['CONTENT'].apply(preprocess_text)

# Basic data checks
print(f"Dataset shape: {df.shape}")
print("Missing values:\n", df.isnull().sum())
print("Unique classes:", df['CLASS'].unique())

# Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=50)

# Split into features (X) and target (y)
X = df_shuffled['CONTENT']
y = df_shuffled['CLASS']

# Split into training (75%) and testing (25%)
train_size = int(0.75 * len(df_shuffled))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Initialize CountVectorizer
count_vectorizer = CountVectorizer()
#count_vectorizer = CountVectorizer(min_df=5, max_features=100)

# Fit and transform the comments
X_train_count_vectorized = count_vectorizer.fit_transform(X_train)

# Display the shape of the transformed data
print("Shape of count vectorized data:", X_train_count_vectorized.shape)

# Display some feature names (words)
print("Sample feature names:", count_vectorizer.get_feature_names_out()[:10])

# Initialize TF-IDF transformer
tfidf_transformer = TfidfTransformer()

# Transform count vectorized data to TF-IDF
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count_vectorized)

# Display the shape of the TF-IDF transformed data
print("Shape of TF-IDF transformed data:", X_train_tfidf.shape)

# Initialize Naive Bayes classifier
model = MultinomialNB()

# Fit the model on training data (using TF-IDF)
model.fit(X_train_tfidf, y_train)

# Cross-validate with 5-folds and print mean accuracy
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print("Mean accuracy from 5-fold cross-validation:", cv_scores.mean())

# Predict on test data
y_pred = model.predict(tfidf_transformer.transform(count_vectorizer.transform(X_test)))

# Confusion Matrix and Accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy on test data:", accuracy)

# Generate new comments for testing
new_comments = [
    "This video is amazing! I love it!",   # Non-spam
    "Great content, keep it up!",           # Non-spam
    "Check out my channel for more videos!", # Spam
    "You should subscribe to my channel!",   # Spam
    "Fantastic tutorial, very informative!",  # Non-spam
    "I really enjoyed this video!",           # Non-spam
    "Make money today, click here", # Spam
    "Your bank account has been blocked, contact me to unlock it", # Spam
    "Come to Turkey", # Spam
    "Keep it up!, I like the choreography", #Non-spam
    "Click here to win an iPhone!", # Spam
    "Don't forget to subscribe for more updates!", # Non-spam
    "This channel helped me become a millionaire. Thanks! [link]", # Spam
    "I followed your advice and got my dream car! Here's how others can too: [link]", # Spam
    "Hot singles near you now! Visit [link]", #Spam
    "I was skeptical, but this is LEGIT! Try it here: [link]", #Spam
    "Your account has been selected for a reward. Click to claim: [link]", # Spam
    "OMG, I can't believe this trick actually works! Check it here: [link]", # Spam
]

# Transform new comments using the same vectorizer and predict
predictions = model.predict(tfidf_transformer.transform(count_vectorizer.transform(new_comments)))

# Display predictions for new comments
for comment, prediction in zip(new_comments, predictions):
    print(f"Comment: '{comment}' - Prediction: {"Spam" if prediction == 1 else "Non-spam"}")