import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text   # IMPORTANT (no src)

print("🚀 Starting training...")

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load data
print("📂 Loading data...")
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real])
df = df[['text', 'label']]
df = df.sample(min(500, len(df)), random_state=42)

print("🧹 Cleaning text...")
df['text'] = df['text'].astype(str).apply(clean_text)
print("Total rows:", len(df))
# Features + Labels
X = df['text']
y = df['label']

print("🔢 Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split
print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
print("💾 Saving model...")
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")