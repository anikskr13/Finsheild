import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_data

DATA_PATH = Path("../data/creditcard.csv")
MODEL_PATH = Path("../models/model.pkl")
SCALER_PATH = Path("../models/scaler.pkl")


def train_model():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model()