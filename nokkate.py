from gender import GenderPredictorHybrid

predictor = GenderPredictorHybrid()
gender, confidence = predictor.predict("Jison Joseph Sebastian")
print(gender)
