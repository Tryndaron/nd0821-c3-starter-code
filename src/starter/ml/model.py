from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from .data import process_data


SLICE_OUTPUT = "/home/roggenlanda/Schreibtisch/projects/Fortbildungen/Machine_learning_devOps/Kurs4_final_project/nd0821-c3-starter-code/starter/model/slice_output.txt"



# Optional: implement hyperparameter tuning.
def train_lr_model(X_train, y_train, max_iter=10000):
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
    lr_model = LogisticRegression(max_iter=max_iter, random_state=42)
    lr_model.fit(X_train, y_train)
    print("Training Successfull !")
    return lr_model






    


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


def compute_model_performance_on_categorical_data(cat_features, model, y_test, y_pred, test_data, encoder, lb):
    """_summary_

    Args:
        cat_features (_type_): _description_
        model (_type_): _description_
        y_test (_type_): _description_
        y_pred (_type_): _description_
        test_data (_type_): _description_
        encoder (_type_): _description_
        lb (_type_): _description_
    """
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    metrics = []
    for cat in cat_features:
        for cat_variation in test_data[cat].unique():
            slice_df = test_data[test_data[cat] == cat_variation]
            X_slice, y_slice, _, _ = process_data(
                slice_df, categorical_features=cat_features,
                label='salary', training=False, encoder=encoder, lb=lb)
            y_slice_pred = model.predict(X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice,
                                                              y_slice_pred)
            metrics.append(f"Category feature: {cat}, Category variation: {cat_variation}, Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
    with open(SLICE_OUTPUT, 'w') as file:
        file.write('\n'.join(metrics))
    
            
              



def inference(model: LogisticRegression , X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Logistice Regression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    predictions = model.predict(X)
    return predictions
