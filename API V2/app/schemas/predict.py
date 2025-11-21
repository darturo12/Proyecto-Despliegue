from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]  # Cambiado a int porque Subscription es 0/1

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Age": 57,
                        "Job": "management",
                        "Marital.Status": "married",
                        "Education": "tertiary",
                        "Credit": "no",
                        "Balance.euros.": 1506,
                        "Housing.Loan": "yes",
                        "Personal.Loan": "no",
                        "Contact": "cellular",
                        "Last.Contact.Day": 19,
                        "Last.Contact.Month": "nov",
                        "Last.Contact.Duration": 79,
                        "Campaign": 1,
                        "Pdays": -1,
                        "Previous": 0,
                        "Poutcome": "unknown"
                    }
                ]
            }
        }