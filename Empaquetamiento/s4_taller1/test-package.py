import os
import pandas as pd
from model.predict import make_prediction
from model.config.core import DATASET_DIR, config

# Cargar el archivo configurado test_data_file
file_path = DATASET_DIR / config.app_config.test_data_file

sample_input_data = pd.read_csv(file_path)

result = make_prediction(input_data=sample_input_data)
print(result)