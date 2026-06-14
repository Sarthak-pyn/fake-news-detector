import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

#load data
print("loading data")
real = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")
#add label 
real["label"] = 1
fake["label"] = 0
# ── 3. COMBINE & SHUFFLE ───────────
df = pd.concat([real, fake]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total articles: {len(df)}")
print(f"Real: {df['label'].sum()} | Fake: {(df['label']==0).sum()}")
# ── 4. COMBINE TITLE + TEXT ──────────────
df["content"] = df["title"] + " " + df["text"]

# ── 5. SPLIT DATA ─────────
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
# ── 6. TF-IDF VECTORIZER ────────
print("\nApplying TF-IDF...")
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ── 7. TRAIN MODEL ──────
print("Training model...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# ── 8. EVALUATE ─────────
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
# ── 9. SAVE MODEL & VECTORIZER ────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("\n✅ model.pkl and tfidf.pkl saved!")
