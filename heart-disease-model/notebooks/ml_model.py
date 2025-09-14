
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
    def __init__(self, dataset: pd.DataFrame, target_column: str, mlflow_url: str, experiment_name: str, input_example: str):
        self.X = dataset.drop(columns=[target_column])
        self.y = dataset[target_column]
        mlflow.set_tracking_uri(mlflow_url)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.input_example = input_example

    def run(self):

        xCombinaciones, xParametros = self.full_models()

        for model, ts in xCombinaciones:

            with mlflow.start_run(run_name=self.experiment_name):

                X_train, X_test, y_train, y_test = train_test_split( self.X, self.y, test_size=ts )

                print(f"Entrenando modelo: {model} con % {ts}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                signature = infer_signature(X_train, model.predict(X_train))

                # Evaluación
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                accuracy = accuracy_score(y_test, y_pred)
                error_rate = 1 - accuracy
                recall = recall_score(y_test, y_pred)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                print(f"    Registrando Experimento")

                mlflow.log_param("1.Modelo_Usado", model.__class__.__name__)
                mlflow.log_param("2.Porcen Valid", ts)

                p, d = model.get_params(False), type(model)().get_params(False)
                changed = {k: p[k] for k in p if p[k] != d.get(k)}
                mlflow.log_param("params", __import__("json").dumps(changed, ensure_ascii=False))
                
                mlflow.log_metric("1.Error_Rate", round(error_rate,5))
                mlflow.log_metric("2.Accuracy", round(accuracy,5) )
                mlflow.log_metric("3.ReCall", round(recall,5) )
                mlflow.log_metric("4.Specificity", round(specificity,5) )

                mlflow.sklearn.log_model(
                    sk_model = model,
                    artifact_path = "model_cardio",
                    signature = signature,
                    input_example = self.input_example
                )

    def full_models(self):

        test_sizes = [0.10]#, 0.15, 0.20]

        logistic_models = [
            #LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000),
            #LogisticRegression(penalty="l2", C=0.5, solver="lbfgs", max_iter=5000),
            #LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000),
            #LogisticRegression(penalty="elasticnet", C=1.0, solver="saga", l1_ratio=0.5, max_iter=5000),
            #LogisticRegression(penalty="l2", C=2.0, solver="saga", max_iter=5000)
        ]

        rf_models = [
            RandomForestClassifier(n_estimators=100, n_jobs=-1),
            #RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1),
            #RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1),
            #RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1),
            RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        ]

        svc_models = [
            #SVC(kernel="linear", C=1.0, probability=True),
            #SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
            #SVC(kernel="poly", degree=3, C=1.0, probability=True),
            #SVC(kernel="rbf", C=0.5, gamma="auto", probability=True),
            #SVC(kernel="sigmoid", C=1.0, probability=True)
            #SVC(kernel="linear", C=0.5),
            #SVC(kernel="rbf", C=0.5, gamma="scale"),
            #SVC(kernel="poly", degree=2, C=0.5),
            #SVC(kernel="rbf", C=0.1, gamma="auto"),
            #SVC(kernel="linear", C=1.0, tol=1e-2)
        ]

        mlp_models = [
            #MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
            #MLPClassifier(hidden_layer_sizes=(100,50), activation="tanh", max_iter=1500),
            #MLPClassifier(hidden_layer_sizes=(128,64,32), activation="relu", max_iter=2000),
            #MLPClassifier(hidden_layer_sizes=(256,128), solver="sgd", learning_rate_init=0.01, max_iter=2000),
            #MLPClassifier(hidden_layer_sizes=(64,64,64), activation="relu", solver="adam", max_iter=2000)
            MLPClassifier(hidden_layer_sizes=(20,), max_iter=500),
            MLPClassifier(hidden_layer_sizes=(32,16), activation="relu", max_iter=800),
            MLPClassifier(hidden_layer_sizes=(40,), activation="tanh", max_iter=700),
            MLPClassifier(hidden_layer_sizes=(30,), solver="sgd", learning_rate_init=0.05, max_iter=600),
            MLPClassifier(hidden_layer_sizes=(16,16,8), activation="relu", max_iter=800)
        ]
      
        modelos = logistic_models + rf_models + svc_models + mlp_models

        combinaciones = [(m, ts) for m in modelos for ts in test_sizes]

        parametros = [
            "penalty", "C", "solver", "max_iter", "l1_ratio",                     # LogisticRegression
            "n_estimators", "max_depth", "max_features", "n_jobs",                # RandomForest
            "kernel", "gamma", "degree", "probability", "tol",                    # SVC
            "hidden_layer_sizes", "activation", "solver", "learning_rate_init"   # MLP
        ]
        
        return combinaciones, parametros
