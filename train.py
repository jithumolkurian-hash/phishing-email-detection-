import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data.csv")

# Split data
X = df["text"]
y = df["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

print("Model trained successfully!")

# Test
#email = ["Click here to win money"]
#email_vec = vectorizer.transform(email)

email=["meet me in the morning"]
email_vec = vectorizer.transform(email)

prediction = model.predict(email_vec)
print("Prediction:", prediction[0])