import numpy as np
from model.predict import make_prediction


def test_make_prediction(sample_input_data):

    result = make_prediction(input_data=sample_input_data)

    predictions = result.get("predictions")
    errors = result.get("errors")

    # Validaciones mínimas
    assert errors is None
    assert isinstance(predictions, list)
    assert len(predictions) > 0  # No necesita valor exacto

    # Validar que el primer valor sea numérico
    first_pred = predictions[0]

    if isinstance(first_pred, np.generic):
        first_pred = float(first_pred)

    assert isinstance(first_pred, (float, int))
