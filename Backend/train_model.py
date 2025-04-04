import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
try:
    df = pd.read_csv('train.csv', encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 decoding failed. Trying 'latin1' encoding...")
    df = pd.read_csv('train.csv', encoding='latin1')

# Step 2: Preprocess the text data
def remove_unnecessary_characters(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

df['clean_text'] = df['text'].apply(remove_unnecessary_characters)

# Step 3: Define features and target
X = df['clean_text']
y = df['sentiment']

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorize the text
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 6: Train Random Forest with GridSearchCV
param_grid = {'n_estimators': [200, 400, 600], 'max_depth': [15, 25, 35]}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_vectorized, y_train)

# Step 7: Evaluate
best_rf_classifier = grid_search.best_estimator_
y_pred = best_rf_classifier.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Step 8: Save model and vectorizer
joblib.dump(best_rf_classifier, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
