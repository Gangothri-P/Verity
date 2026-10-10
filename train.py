
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the datasets
print("Loading datasets...")

fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# 2. Assign labels
fake_df["label"] = 0
real_df["label"] = 1

# 3. Combine both datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

print("\nOriginal dataset shape:", df.shape)

# 4. Remove exact duplicate rows
df = df.drop_duplicates().copy()

print("Shape after duplicate removal:", df.shape)

# 5. Create the text used for prediction
df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
).str.strip()

# Remove rows with empty content
df = df[df["content"] != ""].copy()

# 6. Prepare features and labels
X = df["content"]
y = df["label"]

print("\nDataset label counts:")
print(y.value_counts())

# 7. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. Create the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_df=0.7
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# 9. Train the model
print("\nTraining model... Please wait.")

model.fit(X_train, y_train)

print("Training complete!")

# 10. Evaluate the model
y_pred = model.predict(X_test)

print("\n" + "=" * 45)
print("MODEL EVALUATION")
print("=" * 45)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 11. Save the trained model
Path("models").mkdir(exist_ok=True)

model_path = Path("models/fake_news_model.joblib")
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("\nAll training steps completed successfully!")
