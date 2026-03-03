import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pd.read_csv("student_data.csv")

X = data.drop("result", axis=1)
y = data["result"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")