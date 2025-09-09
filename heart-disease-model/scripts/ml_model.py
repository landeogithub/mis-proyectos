
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class ModelRunner:
    def __init__(self, dataset: pd.DataFrame, target_column: str, mlflow_url: str, experiment_name: str):
        self.X = dataset.drop(columns=[target_column])
        self.y = dataset[target_column]
        mlflow.set_tracking_uri(mlflow_url)
        mlflow.set_experiment(experiment_name)

    def run(self):
        combinaciones = [
            ( LogisticRegression(), 0.10 ),     ( LogisticRegression(), 0.15 ),     ( LogisticRegression(), 0.20 ),
            ( RandomForestClassifier(), 0.10 ), ( RandomForestClassifier(), 0.15 ), ( RandomForestClassifier(), 0.20 ),
            ( SVC(), 0.10 ),                    ( SVC(), 0.15 ),                    ( SVC(), 0.20 ),
            ( MLPClassifier(), 0.10 ),          ( MLPClassifier(), 0.15 ),          ( MLPClassifier(), 0.20 )
        ]

        for model, ts in combinaciones:
            X_train, X_test, y_train, y_test = train_test_split( self.X, self.y, test_size=ts )
            run_name = f"{model.__class__.__name__}_ts{ts}"
            with mlflow.start_run(run_name=run_name):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_param("test_size", ts)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(model, "model")
