
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import mlflow, mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.models import infer_signature

class ModelRunner:
    def __init__(self, dataset: pd.DataFrame, target_column: str, mlflow_url: str, experiment_name: str):
        self.X = dataset.drop(columns=[target_column])
        self.y = dataset[target_column]
        mlflow.set_tracking_uri(mlflow_url)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def run(self):

        test_sizes = [0.10, 0.15, 0.20]

        logistic_models = [
    LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000),
    LogisticRegression(penalty="l2", C=0.5, solver="lbfgs", max_iter=5000),
    LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000),
    LogisticRegression(penalty="elasticnet", C=1.0, solver="saga", l1_ratio=0.5, max_iter=5000),
    LogisticRegression(penalty="l2", C=2.0, solver="saga", max_iter=5000)
]

        rf_models = [
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1),
    RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
]

        svc_models = [
    SVC(kernel="linear", C=1.0, probability=True, random_state=42),
    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=42),
    SVC(kernel="rbf", C=0.5, gamma="auto", probability=True, random_state=42),
    SVC(kernel="sigmoid", C=1.0, probability=True, random_state=42)
]

        mlp_models = [
    MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    MLPClassifier(hidden_layer_sizes=(100,50), activation="tanh", max_iter=1500, random_state=42),
    MLPClassifier(hidden_layer_sizes=(128,64,32), activation="relu", max_iter=2000, random_state=42),
    MLPClassifier(hidden_layer_sizes=(256,128), solver="sgd", learning_rate_init=0.01, max_iter=2000, random_state=42),
    MLPClassifier(hidden_layer_sizes=(64,64,64), activation="relu", solver="adam", max_iter=2000, random_state=42)
]

        modelos = logistic_models + rf_models + svc_models + mlp_models

        combinaciones = [(m, ts) for m in modelos for ts in test_sizes]

        for model, ts in combinaciones:
            X_train, X_test, y_train, y_test = train_test_split( self.X, self.y, test_size=ts )
            #run_name = f"{model.__class__.__name__}_ts{ts}"
            with mlflow.start_run(run_name=self.experiment_name):
                
                print(f"Entrenando modelo: {model} con % {ts}")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluación
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                accuracy = accuracy_score(y_test, y_pred)
                error_rate = 1 - accuracy
                recall = recall_score(y_test, y_pred)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                mlflow.log_param("1.Modelo_Usado", model.__class__.__name__)
                mlflow.log_param("2.Porcen Valid", ts)
                #mlflow.log_param("3.Configuracion", config_str)
                mlflow.log_metric("1.Error_Rate", round(error_rate,5))
                mlflow.log_metric("2.Accuracy", round(accuracy,5) )
                mlflow.log_metric("3.ReCall", round(recall,5) )
                mlflow.log_metric("4.Specificity", round(specificity,5) )

                signature = infer_signature(X_train, model.predict(X_train))
    
                input_example = {
                    "age": 18393, "gender": "2", "height": 168, "weight": 62.0, "ap_hi": 110, "ap_lo": 80,
                    "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1
                }
                
                mlflow.sklearn.log_model(
                    sk_model = model,
                    artifact_path = "model_cardio",
                    signature = signature,
                    input_example = input_example
                )
                




