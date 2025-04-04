#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns

from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')
import os


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[7]:


train_data = pd.read_csv('train.csv',encoding='latin1');
test_data = pd.read_csv('test.csv',encoding='latin1');
df = pd.concat([train_data,test_data])


# In[8]:


df.head()


# In[9]:


def remove_unnecessary_characters(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text
df['clean_text'] = df['text'].apply(remove_unnecessary_characters)


# In[10]:


import nltk
nltk.download('punkt_tab')
def tokenize_text(text):
    try:
        text = str(text)
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return []
df['tokens'] = df['text'].apply(tokenize_text)


# In[11]:


def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = str(text)
    return text
df['normalized_text'] = df['text'].apply(normalize_text)


# In[12]:


import nltk
nltk.download('stopwords')
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
        filtered_text = ' '.join(filtered_words)
    else:
        filtered_text = ''
    return filtered_text
df['text_without_stopwords'] = df['text'].apply(remove_stopwords)


# In[13]:


df.dropna(inplace=True)


# In[14]:


df['sentiment_code'] = df['sentiment'].astype('category').cat.codes
sentiment_distribution = df['sentiment_code'].value_counts(normalize=True)


# In[15]:


print("DataFrame shape:", df.shape)
print("Is DataFrame empty?", df.empty)
print("Columns:", df.columns)
print("Missing values in 'text':", df['text'].isna().sum())

if 'text' in df.columns and not df.empty:
    corpus = df['text'].dropna().tolist()
    print("Corpus length:", len(corpus))
    if corpus:  # Check if list is not empty
        print("First text sample:", corpus[0])
    else:
        print("Corpus is empty after removing NaN values.")
else:
    print("No valid 'text' column found.")


# In[16]:


final_corpus = df['text'].astype(str).tolist()
data_eda = pd.DataFrame()
data_eda['text'] = final_corpus
data_eda['sentiment'] = df["sentiment"].values
data_eda.head()


# In[17]:


# df['Time of Tweet'] = df['Time of Tweet'].astype('category').cat.codes
# df['Country'] = df['Country'].astype('category').cat.codes
# df['Age of User']=df['Age of User'].replace({'0-20':18,'21-30':25,'31-45':38,'46-60':53,'60-70':65,'70-100':80})


# In[18]:


df=df.drop(columns=['textID','Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'])


# In[19]:


import string

def wp(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Assuming df is defined somewhere in your code
df['selected_text'] = df["selected_text"].apply(wp)


# In[20]:


X=df['selected_text']
y= df['sentiment']


# In[21]:


import pandas as pd

# Load the dataset with a specific encoding
try:
    df = pd.read_csv('train.csv', encoding='utf-8')  # Try UTF-8 first
except UnicodeDecodeError:
    print("UTF-8 decoding failed. Trying 'latin1' encoding...")
    df = pd.read_csv('train.csv', encoding='latin1')  # Fallback to latin1

# Display basic information
print("Dataset shape:", df.shape)
print("Column names:", df.columns)
print(df.head())

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values if necessary
df = df.dropna()

# Verify the cleaned dataset
print("Cleaned dataset shape:", df.shape)


# In[25]:


from sklearn.model_selection import  GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[26]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
try:
    df = pd.read_csv('train.csv', encoding='utf-8')  # Try UTF-8 first
except UnicodeDecodeError:
    print("UTF-8 decoding failed. Trying 'latin1' encoding...")
    df = pd.read_csv('train.csv', encoding='latin1')  # Fallback to latin1

# Step 2: Inspect the dataset
print("Dataset shape:", df.shape)
print("Column names:", df.columns)
print(df.head())

# Step 3: Preprocess the text data
def remove_unnecessary_characters(text):
    if pd.isna(text):  # Handle NaN or None values
        return ''
    text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))  # Keep only alphanumeric and spaces
    text = re.sub(r'\s+', ' ', str(text)).strip()  # Replace multiple spaces with a single space and trim
    return text

df['clean_text'] = df['text'].apply(remove_unnecessary_characters)

# Step 4: Define X (features) and y (target)
X = df['clean_text']  # Use the cleaned text as features
y = df['sentiment']   # Use the 'sentiment' column as the target

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Text Vectorization with trigram support
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 7: Define the Random Forest classifier with expanded hyperparameter search
param_grid = {'n_estimators': [200, 400, 600], 'max_depth': [15, 25, 35]}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_vectorized, y_train)

# Step 8: Make predictions on the test set using the best estimator from grid search
best_rf_classifier = grid_search.best_estimator_
y_pred = best_rf_classifier.predict(X_test_vectorized)

# Step 9: Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# In[27]:


import joblib
joblib.dump(best_rf_classifier, 'sentiment_model.pkl')


# In[ ]:


# loaded_model = joblib.load('sentiment_model.pkl')

# new_text = "i hate u"

# new_text_vectorized = vectorizer.transform([new_text])

# predicted_sentiment = loaded_model.predict(new_text_vectorized)

# print(f'Predicted Sentiment: {predicted_sentiment[0]}')



def analyze_text(text):
    print("Loading model...")
    loaded_model = joblib.load('sentiment_model.pkl')

    print("Loading vectorizer...")
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("Vectorizer loaded successfully!")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return "Vectorizer loading failed."

    print(f"Vectorizing text: {text}")
    try:
        text_vectorized = vectorizer.transform([text])
        print("Text successfully vectorized.")
    except Exception as e:
        print(f"Error vectorizing text: {e}")
        return "Vectorization failed."

    print("Predicting sentiment...")
    predicted_sentiment = loaded_model.predict(text_vectorized)

    print("Prediction complete.")
    return predicted_sentiment[0]


# In[29]:


# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Step 1: Load the datasets
# try:
#     train_data = pd.read_csv('train.csv', encoding='utf-8')  # Try UTF-8 first
# except UnicodeDecodeError:
#     print("UTF-8 decoding failed for train.csv. Trying 'latin1' encoding...")
#     train_data = pd.read_csv('train.csv', encoding='latin1')  # Fallback to latin1

# try:
#     test_data = pd.read_csv('test.csv', encoding='utf-8')  # Try UTF-8 first
# except UnicodeDecodeError:
#     print("UTF-8 decoding failed for test.csv. Trying 'latin1' encoding...")
#     test_data = pd.read_csv('test.csv', encoding='latin1')  # Fallback to latin1

# # Step 2: Inspect the datasets
# print("Training dataset shape:", train_data.shape)
# print("Testing dataset shape:", test_data.shape)
# print("Training dataset columns:", train_data.columns)
# print("Testing dataset columns:", test_data.columns)

# # Step 3: Preprocess the text data
# def remove_unnecessary_characters(text):
#     if pd.isna(text):  # Handle NaN or None values
#         return ''
#     text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))  # Keep only alphanumeric and spaces
#     text = re.sub(r'\s+', ' ', str(text)).strip()  # Replace multiple spaces with a single space and trim
#     return text

# # Apply preprocessing to both training and testing datasets
# train_data['clean_text'] = train_data['text'].apply(remove_unnecessary_characters)
# test_data['clean_text'] = test_data['text'].apply(remove_unnecessary_characters)

# # Step 4: Define features (X) and target (y) for training
# X_train = train_data['clean_text']
# y_train = train_data['sentiment']

# # For testing, we only have features (no target labels in test.csv)
# X_test = test_data['clean_text']

# # Step 5: Vectorize the text data using TF-IDF with trigram support
# vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# # Step 6: Split the training data into training and validation sets
# X_train_final, X_val, y_train_final, y_val = train_test_split(
#     X_train_vectorized, y_train, test_size=0.2, random_state=42
# )

# # Step 7: Train a Random Forest classifier
# rf_classifier = RandomForestClassifier(random_state=42)
# rf_classifier.fit(X_train_final, y_train_final)

# # Step 8: Evaluate the model on the validation set
# y_val_pred = rf_classifier.predict(X_val)
# validation_accuracy = accuracy_score(y_val, y_val_pred)
# print(f'Validation Accuracy: {validation_accuracy:.2f}')

# # Step 9: Make predictions on the test set
# test_predictions = rf_classifier.predict(X_test_vectorized)

# # Step 10: Save the predictions to a CSV file (if needed)
# test_data['predicted_sentiment'] = test_predictions
# test_data[['textID', 'predicted_sentiment']].to_csv('submission.csv', index=False)

# print("Test set predictions saved to 'submission.csv'")

