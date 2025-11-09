#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


mlflow.set_tracking_uri("file:/home/ubuntu/Proyecto-Despliegue/mlruns")
mlflow.set_experiment("Modelo de clasificación - XGBoost_Suscripcion")


def run_mlflow(run_name="Entrenamiento_XGBoost"):
    with mlflow.start_run(run_name=run_name) as run:
        # IDs de referencia
        experimentID = run.info.experiment_id
        runID = run.info.run_id

        # 1. Cargar datos
        train = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vS60HQztm8Bh9VS0LiK9Uhr9llcpJnxpefPrNKMC7TjWY3PslbhPBLDuc2Pf2t-yzwO6nXOPGpHSdr-/pub?output=csv")
        test = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTuZODZ6QTsYqzReqNm9arsBIc-tusb063SPYCB6riVdpbezq0lZs3mWDIvYVOodjvM4zPe-QekUS5Q/pub?output=csv")

        y = train['Subscription']
        X = train.drop(columns=['Subscription'])

        # 2. Identificar tipos de variables
        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(include='object').columns.tolist()

        # 3. Preprocesamiento
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

        # 4. Pipeline con modelo
        clf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ))
        ])

        # 5. División de datos
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # 6. Búsqueda de hiperparámetros
        param_grid = {
            'classifier__n_estimators': [600, 900, 1200, 1500],
            'classifier__max_depth': [2, 4, 6],
            'classifier__learning_rate': [0.05, 0.1, 0.25],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }

        random_search = RandomizedSearchCV(
            clf_pipeline,
            param_distributions=param_grid,
            n_iter=20,
            scoring='roc_auc',
            cv=5,
            verbose=1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # 7. Log de parámetros y métricas
        mlflow.log_params(random_search.best_params_)

        y_pred_proba = best_model.predict_proba(X_valid)[:, 1]
        y_pred = best_model.predict(X_valid)

        metrics = {
            "AUC_ROC": roc_auc_score(y_valid, y_pred_proba),
            "Accuracy": accuracy_score(y_valid, y_pred),
            "Precision": precision_score(y_valid, y_pred),
            "Recall": recall_score(y_valid, y_pred),
            "F1": f1_score(y_valid, y_pred)
        }

        mlflow.log_metrics(metrics)

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
        plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC_ROC']:.4f}")
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
        cm = confusion_matrix(y_valid, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot(cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 11. Guardar modelo completo (pipeline)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="modelo_xgb_pipeline",
            registered_model_name="Modelo_ClasificacionV2"
        )

        print(f"\nMLflow Run completado con run_id {runID} y experiment_id {experimentID}")
        return experimentID, runID


if __name__ == "__main__":
    expID, runID = run_mlflow()
    print(f"Ejecutado correctamente en experimento ID: {expID}")
