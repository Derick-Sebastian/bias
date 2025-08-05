import json
import numpy as np
from decission_model.hireml import HiringPredictor
from decission_model.resumescore import ResumeComparator
from decission_model.Resumeconverter import ResumeConverter
from decission_model.namepars import ResumeNameExtractor
from decission_model.gender import GenderPredictorHybrid

class CandidateEvaluator:
    def __init__(self, file_path, interview_score, test_score, hiring_model_path="hiring_model.pkl"):
        self.file_path = file_path
        self.interview_score = interview_score
        self.test_score = test_score
        self.hiring_model_path = hiring_model_path

    def run(self):
        result = {}

        # Extract name
        name_extractor = ResumeNameExtractor(self.file_path)
        name = name_extractor.extract_name()
        result['name'] = name

        # Predict gender
        gender_predictor = GenderPredictorHybrid()
        gender, gender_confidence = gender_predictor.predict(name)
        result['gender'] = gender

        # Convert resume to plain text
        resume_converter = ResumeConverter(self.file_path)
        resume_text = resume_converter.convert_to_txt()

        # Resume similarity score
        comparator = ResumeComparator()
        similarity_score = comparator.compare_with_text(resume_text)
        result['resume_similarity_score'] = round(float(similarity_score), 2)

        # Hiring decision
        hiring_predictor = HiringPredictor(self.hiring_model_path)
        prediction, prediction_confidence = hiring_predictor.predict(
            float(similarity_score),
            int(self.interview_score),
            int(self.test_score)
        )
        result['hiring_decision'] = str(prediction)
        result['hiring_confidence'] = round(float(prediction_confidence) * 100, 2)

        # 🔁 Convert all values to native types
        result = {k: (float(v) if isinstance(v, (np.float32, np.float64))
                      else int(v) if isinstance(v, (np.int32, np.int64))
                      else v)
                  for k, v in result.items()}

        return json.dumps(result, indent=4)
