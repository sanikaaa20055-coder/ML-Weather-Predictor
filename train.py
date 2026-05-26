import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import pickle

# LOAD DATASET

df = pd.read_csv(
    "weather_dataset.csv"
)

# REMOVE NULL VALUES

df.dropna(inplace=True)

# FEATURES

X = df[[

    "Temp_C",
    "Rel Hum_%",
    "Wind Speed_km/h",
    "Press_kPa"

]]

# TARGET

y = df["Weather"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# RANDOM FOREST MODEL

model = RandomForestClassifier()

# TRAIN MODEL

model.fit(
    X_train,
    y_train
)

# SAVE MODEL

pickle.dump(
    model,
    open("model.pkl", "wb")
)

print(
    "Model Trained Successfully"
)
