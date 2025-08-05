import joblib
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

class HiringPredictor:
    def __init__(self, model_path = "hiring_model.pkl"
):
        # Suppress version mismatch warning and other UserWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            warnings.simplefilter("ignore", UserWarning)
            self.model = joblib.load(model_path)

    def predict(self, resume_score, interview_score, test_score):
        features = np.array([[resume_score, interview_score, test_score]])
        
        # Suppress feature name warnings when predicting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0][prediction]
        
        return prediction, round(confidence, 4)