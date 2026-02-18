from typing import List, Optional, Sequence, Tuple, Union

from tensorflow.keras.models import load_model
import pickle

import joblib
import numpy as np


class utils:

    def __init__(self):
        self.model = None
        self.scaler = None  # scaler for features (X)
        
    def load(self) -> None:
        try:
            self.model = load_model("b3_lstm_model.keras")
            # self.model.eval()    
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        try:
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed loading scaler: {e}")  
            
    def predict(self, PredictRequest) -> dict:
        PredictRequest = PredictRequest.sequence
        PredictRequest = np.array(PredictRequest)

        print('scaler')
        print(PredictRequest)

        try:
            scaled_data = self.scaler.transform(PredictRequest.reshape(-1, 1))
        except Exception as e:
            raise RuntimeError(f"Error in scaling: {e}")

        scaled_data = np.array(scaled_data)          # (60, 1)
        scaled_data = scaled_data.reshape(1, 60, 1)   # ✅ (1, 60, 1)

        print('predictor')
        print(scaled_data)

        try: 
            prediction_s = self.model.predict(scaled_data)
        except Exception as e:
            raise RuntimeError(f"Error in predicting: {e}")

        print('des inverser')
        print(prediction_s)
            
        try:
            prediction = self.scaler.inverse_transform(prediction_s)
            return {"prediction": prediction[0][0]}
        except Exception as e:
            raise RuntimeError(f"Error in inverse scaling: {e}")