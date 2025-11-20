import typing as t

import numpy as np
import pandas as pd

from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import load_pipeline
from model.processing.validation import validate_inputs


# Nombre del archivo del pipeline
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"

# Cargar el pipeline entrenado
_modelo_suscripcion_bancaria = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    # Convertir a DataFrame
    data = pd.DataFrame(input_data)

    # Validar entradas
    validated_data, errors = validate_inputs(input_data=data)

    # Resultado base
    results = {"predictions": None, "version": _version, "errors": errors}

    # Si no hay errores → predecir
    if not errors:
        predictions = _modelo_suscripcion_bancaria.predict(
            X=validated_data[config.model_config.features]
        )
        
        results = {
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results

