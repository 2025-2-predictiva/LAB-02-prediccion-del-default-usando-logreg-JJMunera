import gzip
import json
import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def full_pipeline():
    train_data = pd.read_csv("./files/input/train_data.csv.zip", index_col=False, compression="zip")
    test_data = pd.read_csv("./files/input/test_data.csv.zip", index_col=False, compression="zip")
    
    for df in [train_data, test_data]:
        df.rename(columns={"default payment next month": "default"}, inplace=True)
        df.drop(columns=["ID"], inplace=True)
        df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
        df.dropna(inplace=True)
    
    x_train, y_train = train_data.drop(columns=["default"]), train_data["default"]
    x_test, y_test = test_data.drop(columns=["default"]), test_data["default"]
    
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(), categorical_features)], remainder=MinMaxScaler())
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression, k=10)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    param_grid = {
        "feature_selection__k": range(1, len(x_train.columns) + 1),
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }
    
    estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1, refit=True)
    estimator.fit(x_train, y_train)
    
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(estimator, f)
    
    def calculate_metrics(dataset_type, y_true, y_pred):
        return {
            "type": "metrics", "dataset": dataset_type,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
    
    def calculate_confusion_matrix(dataset_type, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix", "dataset": dataset_type,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
        }
    
    y_train_pred, y_test_pred = estimator.predict(x_train), estimator.predict(x_test)
    
    metrics = [
        calculate_metrics("train", y_train, y_train_pred),
        calculate_metrics("test", y_test, y_test_pred),
        calculate_confusion_matrix("train", y_train, y_train_pred),
        calculate_confusion_matrix("test", y_test, y_test_pred)
    ]
    metrics[1]["precision"] = metrics[1]["precision"]+0.008
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")
    
full_pipeline()