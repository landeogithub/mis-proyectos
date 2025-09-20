
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

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


                #  ColumnTransformer automático
                preprocessor = ColumnTransformer([
                        (  # Varianbles Categóricas
                            "target_enc",
                            ce.TargetEncoder(),
                            make_column_selector(dtype_include=object)
                        ),  
                        (   # Variables Numéricas
                            "scaler",
                            StandardScaler(),
                            make_column_selector(dtype_include="number")
                        )     
                ])

                #  Pipeline final
                pipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", LogisticRegression())
                ])
                
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size = ts
                )

                print(f"Entrenando modelo: {model} con % {ts}")
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                signature = infer_signature(X_train, pipeline.predict(X_train))

                # Evaluación
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                accuracy = accuracy_score(y_test, y_pred)
                error_rate = 1 - accuracy
                recall = recall_score(y_test, y_pred)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                print(f"    Registrando Experimento")

                mlflow.log_param("1.Modelo Usado", model.__class__.__name__)
                mlflow.log_param("2.Porcentaje Valido", ts)

                p, d = model.get_params(False), type(model)().get_params(False)
                changed = {k: p[k] for k in p if p[k] != d.get(k)}
                #mlflow.log_param("1.Parametros", __import__("json").dumps(changed, ensure_ascii=False))
                mlflow.log_param("1.Parametros", __import__("json").dumps(changed, ensure_ascii=False, default=str))

                
                mlflow.log_metric("1.Error Rate", round(error_rate,5))
                mlflow.log_metric("2.Accuracy", round(accuracy,5) )
                mlflow.log_metric("3.ReCall", round(recall,5) )
                mlflow.log_metric("4.Specificity", round(specificity,5) )

                mlflow.sklearn.log_model(
                    sk_model = pipeline,
                    artifact_path = "model_cardio",
                    signature = signature,
                    input_example = self.input_example
                )

    def full_models(self):

        test_sizes = [0.10, 0.15, 0.20]

        # --- LOGISTIC REGRESSION (6) ---
        logistic_models = [
            LogisticRegression(penalty="l2",        solver="lbfgs", C=0.25, max_iter=20000, tol=1e-6),
            LogisticRegression(penalty="l2",        solver="lbfgs", C=1.0,  max_iter=30000, tol=1e-6),
            LogisticRegression(penalty="l1",        solver="saga",  C=0.5,  max_iter=40000, tol=1e-6),
            LogisticRegression(penalty="l1",        solver="saga",  C=1.0,  max_iter=40000, tol=1e-6),
            LogisticRegression(penalty="elasticnet",solver="saga",  C=1.0,  l1_ratio=0.15, max_iter=40000, tol=1e-6),
            LogisticRegression(penalty="elasticnet",solver="saga",  C=1.5,  l1_ratio=0.85, max_iter=30000, tol=1e-6),
        ]
        
        # --- RANDOM FOREST (6) ---
        rf_models = [
            RandomForestClassifier(n_estimators=800,  max_depth=None, max_features="sqrt", min_samples_split=2,
                                   min_samples_leaf=1, bootstrap=True, oob_score=True, n_jobs=-1),
            RandomForestClassifier(n_estimators=1200, max_depth=24,   max_features="sqrt", min_samples_split=2,
                                   min_samples_leaf=1, bootstrap=True, n_jobs=-1),
            RandomForestClassifier(n_estimators=1000, max_depth=16,   max_features="log2", min_samples_split=5,
                                   min_samples_leaf=2, bootstrap=True, n_jobs=-1),
            RandomForestClassifier(n_estimators=1500, max_depth=None, max_features=0.4,   min_samples_split=10,
                                   min_samples_leaf=2, bootstrap=True, n_jobs=-1),
            RandomForestClassifier(n_estimators=900,  max_depth=30,   max_features="sqrt", min_samples_split=2,
                                   min_samples_leaf=1, bootstrap=True, n_jobs=-1),
            RandomForestClassifier(n_estimators=1200, max_depth=None, max_features="sqrt", min_samples_split=2,
                                   min_samples_leaf=4, bootstrap=True, n_jobs=-1),
        ]
        
        # --- SVC (6) ---
        svc_models = [
            SVC(kernel="rbf",    C=1.0, gamma="scale", probability=False, tol=1e-5),
            SVC(kernel="rbf",    C=2.0, gamma="scale", probability=False, tol=1e-6),
            SVC(kernel="rbf",    C=5.0, gamma="auto",  probability=False, tol=1e-6),
            SVC(kernel="rbf",    C=1.0, gamma=0.01,    probability=False, tol=1e-6),
            SVC(kernel="rbf",    C=2.0, gamma=0.005,   probability=False, tol=1e-6),
            SVC(kernel="linear", C=5.0,                probability=False, tol=1e-6),
        ]
        
        # --- MLP (6) ---
        mlp_models = [
            MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", solver="adam",
                          alpha=1e-4, learning_rate_init=3e-4, max_iter=8000,  early_stopping=True,
                          validation_fraction=0.15, n_iter_no_change=30, tol=1e-5),
            MLPClassifier(hidden_layer_sizes=(512,256),    activation="relu", solver="adam",
                          alpha=1e-3, learning_rate_init=1e-4, max_iter=12000, early_stopping=True,
                          validation_fraction=0.15, n_iter_no_change=40, tol=1e-5),
            MLPClassifier(hidden_layer_sizes=(256,128),    activation="tanh", solver="adam",
                          alpha=1e-4, learning_rate_init=1e-4, max_iter=12000, early_stopping=False, tol=1e-6),
            MLPClassifier(hidden_layer_sizes=(128,64,32),  activation="relu", solver="adam",
                          alpha=1e-5, learning_rate_init=5e-5, max_iter=15000, early_stopping=False, tol=1e-6),
            MLPClassifier(hidden_layer_sizes=(256,),       activation="relu", solver="lbfgs", alpha=1e-4, max_iter=20000, tol=1e-7),
            MLPClassifier(hidden_layer_sizes=(128,128,64), activation="relu", solver="adam",
                          alpha=1e-3, learning_rate_init=3e-4, max_iter=12000, early_stopping=True,
                          validation_fraction=0.2,  n_iter_no_change=50, tol=1e-5),
        ]
        
        # --- DECISION TREE (6) ---
        dt_models = [
            DecisionTreeClassifier(max_depth=None, criterion="gini",    min_samples_split=2,  min_samples_leaf=1,  max_features=None),
            DecisionTreeClassifier(max_depth=24,   criterion="entropy", min_samples_split=5,  min_samples_leaf=2,  max_features="sqrt"),
            DecisionTreeClassifier(max_depth=32,   criterion="gini",    min_samples_split=10, min_samples_leaf=2,  max_features="log2"),
            DecisionTreeClassifier(max_depth=16,   criterion="gini",    min_samples_split=2,  min_samples_leaf=1,  max_features=0.8),
            DecisionTreeClassifier(max_depth=None, criterion="entropy", min_samples_split=2,  min_samples_leaf=4,  max_features=None,
                                   ccp_alpha=1e-4),
            DecisionTreeClassifier(max_depth=20,   criterion="gini",    min_samples_split=5,  min_samples_leaf=2,  max_features="sqrt",
                                   ccp_alpha=5e-4),
        ]
        
        # --- GRADIENT BOOSTING (6) ---
        gb_models = [
            GradientBoostingClassifier(n_estimators=1200, learning_rate=0.05, max_depth=3, subsample=0.9, max_features=None),
            GradientBoostingClassifier(n_estimators=1500, learning_rate=0.03, max_depth=3, subsample=0.8, max_features=None),
            GradientBoostingClassifier(n_estimators=1800, learning_rate=0.02, max_depth=3, subsample=0.7, max_features="sqrt"),
            GradientBoostingClassifier(n_estimators=1200, learning_rate=0.05, max_depth=4, subsample=0.8, max_features="sqrt"),
            GradientBoostingClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.85, max_features="log2"),
            GradientBoostingClassifier(n_estimators=1000, learning_rate=0.08, max_depth=2, subsample=1.0, max_features=None),
        ]
        
        # --- ADABOOST (6) ---
        ab_models = [
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1500, learning_rate=0.05, algorithm="SAMME.R"),
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=2000, learning_rate=0.03, algorithm="SAMME.R"),
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=1200, learning_rate=0.05, algorithm="SAMME.R"),
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=1600, learning_rate=0.04, algorithm="SAMME.R"),
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=2500, learning_rate=0.02, algorithm="SAMME.R"),
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=1200, learning_rate=0.08, algorithm="SAMME.R"),
        ]

        modelos = logistic_models + rf_models + svc_models + mlp_models + dt_models + gb_models + ab_models

        combinaciones = [(m, ts) for m in modelos for ts in test_sizes]

        parametros = [
            # General
            "max_iter", "class_weight",
        
            # Regularización / optimización
            "penalty", "C", "solver", "l1_ratio", "tol", "alpha",
        
            # Random Forest / Decision Tree
            "n_estimators", "max_depth", "max_features", "min_samples_split",
            "min_samples_leaf", "bootstrap", "oob_score",
        
            # SVC
            "kernel", "gamma", "degree", "probability",
        
            # MLP
            "hidden_layer_sizes", "activation", "learning_rate_init",
            "early_stopping", "batch_size",
        
            # Decision Tree
            "criterion", "splitter",
        
            # Gradient Boosting
            "learning_rate", "subsample",
        
            # AdaBoost
            "algorithm", "estimator"
        ]
        
        return combinaciones, parametros
