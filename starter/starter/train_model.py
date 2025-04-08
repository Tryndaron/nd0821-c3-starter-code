# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
from .ml.model import train_lr_model, compute_model_performance_on_categorical_data
from .ml.data import process_data
import pickle

# Add code to load in the data.

#df = pd.read_csv("../data/census.csv")



# Optional enhancement, use K-fold cross validation instead of a train-test split.
#train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
""" X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
) """

# Proces the test data with the process_data function.

# Train and save a model.
def train_save_model():
    data_file_path = '../data/census.csv'
    data = pd.read_csv(data_file_path)
    print(data)
    train, test = train_test_split(data, test_size=0.20, stratify= data['salary'])
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label='salary', training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)
    
    #train model
    lr_model = train_lr_model(X_train, y_train)

    #save model and metrics
    model_path = '/home/roggenlanda/Schreibtisch/projects/Fortbildungen/Machine_learning_devOps/Kurs4_final_project/nd0821-c3-starter-code/starter/model/lr_model.pkl'
    encoder_path = '/home/roggenlanda/Schreibtisch/projects/Fortbildungen/Machine_learning_devOps/Kurs4_final_project/nd0821-c3-starter-code/starter/model/encoder_path.pkl'
    lb_path = '/home/roggenlanda/Schreibtisch/projects/Fortbildungen/Machine_learning_devOps/Kurs4_final_project/nd0821-c3-starter-code/starter/model/lb_path.pkl'
    with open(model_path, "wb") as model_file:
        pickle.dump(lr_model, model_file)

    with open(lb_path, "wb") as lb_file:
        pickle.dump(lb, lb_file)

    with open(encoder_path, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)
    
    y_pred = lr_model.predict(X_test)
    print(y_pred)
    compute_model_performance_on_categorical_data(cat_features, lr_model, y_test, y_pred, test, encoder, lb)
    #joblib.dump(lr_model, model_path)
    #joblib.dump(encoder,encoder_path)
    #joblib.dump(lr_model, lb_path)
    print(f"Modell wurde erfolgreich gespeichert unter {model_path}")


if __name__ == '__main__':
    train_save_model()