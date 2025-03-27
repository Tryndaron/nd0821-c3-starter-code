from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model_type="classifier", n_estimators=100, random_state=42):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model_type (str):
    Returns
    -------
    model
        Trained machine learning model.
    """
    if model_type=="classifier":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_type == "regressor":
        model=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    else:
        raise ValueError("Ungültiger model_type. Wähle 'classifier' oder 'regressor'.")
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if not isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        raise ValueError("Das übergebene Modell ist kein gültiges RandomForrest Model")
    predictions = model.predict(X)
    return predictions
