from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Remove duplicates, split the dataset, and scale Time and Amount.
    """

    # Remove duplicate transactions
    df = df.drop_duplicates().copy()

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale Time and Amount
    scaler = StandardScaler()

    scale_cols = ["Time", "Amount"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test, scaler