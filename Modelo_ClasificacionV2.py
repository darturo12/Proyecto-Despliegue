#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

os.makedirs("/home/ubuntu/mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:/home/ubuntu/mlruns")


mlflow.set_experiment("Modelo de clasificación")

parser = argparse.ArgumentParser(description='Entrenamiento XGBoost con MLflow')

parser.add_argument('--test_size', '-t', type=float, default=0.3)
parser.add_argument('--n_iter', '-n', type=int, default=20)
parser.add_argument('--cv', '-c', type=int, default=5)
parser.add_argument('--random_state', '-r', type=int, default=42)
parser.add_argument('--experiment_name', '-exp', type=str, default="Modelo de clasificación - XGBoost_Suscripcion")

args = parser.parse_args([])  # usar [] para pruebas locales o None desde terminal


def run_mlflow(run_name="XGB_pipeline_run"):
    # Asegurar experimento activo
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        experimentID = run.info.experiment_id
        runID = run.info.run_id

        # 1. Cargar datos
        train = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vS60HQztm8Bh9VS0LiK9Uhr9llcpJnxpefPrNKMC7TjWY3PslbhPBLDuc2Pf2t-yzwO6nXOPGpHSdr-/pub?output=csv")
        test = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTuZODZ6QTsYqzReqNm9arsBIc-tusb063SPYCB6riVdpbezq0lZs3mWDIvYVOodjvM4zPe-QekUS5Q/pub?output=csv")

        y = train['Subscription']
        X = train.drop(columns=['Subscription'])
        id_test = test.iloc[:, 0]
        X_test = test.copy()

        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(include='object').columns.tolist()

        # 2. Preprocesamiento
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

        # 3. Pipeline del modelo
        clf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', random_state=args.random_state
            ))
        ])

        # 4. División de datos
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
        )

        # 5. Búsqueda aleatoria
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.7, 0.9, 1.0],
            'classifier__colsample_bytree': [0.7, 0.9, 1.0]
        }

        random_search = RandomizedSearchCV(
            clf_pipeline, param_distributions=param_grid,
            n_iter=args.n_iter, scoring='roc_auc', cv=args.cv, verbose=2,
            random_state=args.random_state
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # 6. Log de parámetros
        mlflow.log_params(random_search.best_params_)

        # 7. Evaluación
        y_pred_proba = best_model.predict_proba(X_valid)[:, 1]
        y_pred = best_model.predict(X_valid)

        auc = roc_auc_score(y_valid, y_pred_proba)
        accuracy = accuracy_score(y_valid, y_pred)
        precision = precision_score(y_valid, y_pred)
        recall = recall_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        cm = confusion_matrix(y_valid, y_pred)

        mlflow.log_metric("AUC_ROC", auc)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)

        # 8. Importancia de variables
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(8,6))
        feat_importance.head(10).plot(kind='barh', color='skyblue')
        plt.title("Top 10 variables más importantes")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # 9. Curva ROC
        fpr, tpr, _ = roc_curve(y_valid, y_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Tasa Falsos Positivos')
        plt.ylabel('Tasa Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # 10. Matriz de confusión
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot(cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 11. Guardar modelo
        mlflow.sklearn.log_model(best_model, "modelo_xgb_pipeline")

        mlflow.end_run(status='FINISHED')
        return (experimentID, runID)


if __name__ == "__main__":
    expID, runID = run_mlflow()
    print(f"MLflow Run completed with run_id {runID} and experiment_id {expID}")
