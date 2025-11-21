import numpy as np
from config.core import config
from pipeline import modelo_suscripcion_bancaria
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # 1. Cargar datos
    data = load_dataset(file_name=config.app_config.train_data_file)

    # 2. Separar X e y
    X = data[config.model_config.features]
    y = data[config.model_config.target]   # Subscription (0/1)

    # 3. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # 4. Entrenar pipeline
    modelo_suscripcion_bancaria.fit(X_train, y_train)

    # 5. Guardar pipeline entrenado
    save_pipeline(pipeline_to_persist=modelo_suscripcion_bancaria)


if __name__ == "__main__":
    run_training()
