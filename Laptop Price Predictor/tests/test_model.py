import pytest
import joblib
import pandas as pd
import os

@pytest.fixture
def model():
    model_path = os.path.join('models', 'laptop_pipeline.joblib')
    return joblib.load(model_path)

def test_model_prediction_shape(model):
    test_input = pd.DataFrame([[8, 256, 1.4, 13.3]], 
                               columns=['ram', 'ssd', 'weight', 'screen'])
    prediction = model.predict(test_input)
    assert len(prediction) == 1

def test_price_logic(model):
    weak_laptop = pd.DataFrame([[4, 128, 1.5, 14.0]], 
                                columns=['ram', 'ssd', 'weight', 'screen'])
    strong_laptop = pd.DataFrame([[32, 1024, 1.2, 14.0]], 
                                  columns=['ram', 'ssd', 'weight', 'screen'])
    
    price_weak = model.predict(weak_laptop)[0]
    price_strong = model.predict(strong_laptop)[0]
    
    assert price_strong > price_weak

def test_invalid_input_fails(model):
    with pytest.raises(Exception):
        invalid_input = pd.DataFrame([[8, 256]], columns=['ram', 'ssd'])
        model.predict(invalid_input)