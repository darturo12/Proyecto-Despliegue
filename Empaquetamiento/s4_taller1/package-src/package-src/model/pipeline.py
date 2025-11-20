import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from model.config.core import config

# -------------------------------------------------------------------------
# Identificación de variables numéricas y categóricas
# -------------------------------------------------------------------------

num_features = [
    "Age",
    "Balance.euros.",
    "Last.Contact.Day",
    "Last.Contact.Duration",
    "Campaign",
    "Pdays",
    "Previous"
]

cat_features = [
    "Job",
    "Marital.Status",
    "Education",
    "Credit",
    "Housing.Loan",
    "Personal.Loan",
    "Contact",
    "Last.Contact.Month",
    "Poutcome"
]

# -------------------------------------------------------------------------
# Preprocesamiento numérico
# -------------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# -------------------------------------------------------------------------
# Preprocesamiento categórico
# -------------------------------------------------------------------------
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# -------------------------------------------------------------------------
# ColumnTransformer para unir procesamientos
# -------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)


modelo_suscripcion_bancaria = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier",
            XGBClassifier(
                n_estimators=config.model_config.n_estimators,
                colsample_bytree=config.model_config.colsample_bytree,
                learning_rate=config.model_config.learning_rate,
                subsample=config.model_config.subsample,
                max_depth=config.model_config.max_depth,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=config.model_config.random_state,
            )
        ),
    ]
)


