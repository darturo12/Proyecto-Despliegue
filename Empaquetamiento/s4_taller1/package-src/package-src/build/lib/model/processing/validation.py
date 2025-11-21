from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NA in required model features."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Validate and sanitize inputs using Pydantic."""
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # Replace numpy NaN for validation
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    Age: Optional[int] = Field(alias="Age")
    Job: Optional[str] = Field(alias="Job")
    Marital_Status: Optional[str] = Field(alias="Marital.Status")
    Education: Optional[str] = Field(alias="Education")
    Credit: Optional[str] = Field(alias="Credit")
    Balance_euros: Optional[float] = Field(alias="Balance..euros.")
    Housing_Loan: Optional[str] = Field(alias="Housing.Loan")
    Personal_Loan: Optional[str] = Field(alias="Personal.Loan")
    Contact: Optional[str] = Field(alias="Contact")
    Last_Contact_Day: Optional[int] = Field(alias="Last.Contact.Day")
    Last_Contact_Month: Optional[str] = Field(alias="Last.Contact.Month")
    Last_Contact_Duration: Optional[int] = Field(alias="Last.Contact.Duration")
    Campaign: Optional[int] = Field(alias="Campaign")
    Pdays: Optional[int] = Field(alias="Pdays")
    Previous: Optional[int] = Field(alias="Previous")
    Poutcome: Optional[str] = Field(alias="Poutcome")

    class Config:
        allow_population_by_field_name = True  # permits using alias in input


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
