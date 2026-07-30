import numpy as np
from src.model import create_model

def test_model_output_shape():
    model = create_model()
    input_data = np.random.rand(1, 224, 224, 3)  # mimics a single image input with shape (1, 224, 224, 3)
    output = model.predict(input_data)
    assert output.shape == (1, 1)  