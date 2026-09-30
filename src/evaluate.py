import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocess import preprocess_data

DATA_PATH = Path("../data/creditcard.csv")
MODEL_PATH = Path("../models/model.pkl")


def evaluate_model():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    evaluate_model()