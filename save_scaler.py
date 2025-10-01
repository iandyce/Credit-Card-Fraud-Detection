# save_scaler.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# load data
df = pd.read_csv("data/creditcard.csv")

# features (same as used in training)
X = df.drop("Class", axis=1)

# fit scaler on the dataset (matches what your script did earlier)
scaler = StandardScaler().fit(X)

# save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl created in", os.path.abspath("scaler.pkl"))
