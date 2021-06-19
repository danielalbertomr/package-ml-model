import pytest
import os
from ie_bike_model.model import train_and_persist, predict

def test_train_and_persist_creates_file():
    if os.path.isfile('model.joblib'):
        os.remove('model.joblib')
    
    train_and_persist()
    
    assert os.path.isfile('model.joblib')
    
def test_predict_returns_int():
    dteday="2010-03-03"
    weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy" 
    temp=0.3
    atemp=0.3
    hum=0.8
    hr=10
    windspeed=0.0
    
    result = predict(dteday=dteday, weathersit=weathersit, temp=temp, atemp=atemp, hum=hum, hr=hr, windspeed=windspeed)
    
    assert (isinstance(result, int) and result > 0)