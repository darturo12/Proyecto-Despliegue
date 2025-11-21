import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    
    # Verificar estructura de respuesta
    assert "predictions" in prediction_data
    assert "errors" in prediction_data
    assert "version" in prediction_data
    
    # Verificar predicciones
    assert prediction_data["predictions"] is not None
    assert prediction_data["errors"] is None
    assert len(prediction_data["predictions"]) > 0
    
    # Verificar que todas las predicciones son 0 o 1
    for pred in prediction_data["predictions"]:
        assert pred in [0, 1], f"Predicción inválida: {pred}"