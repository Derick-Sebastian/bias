import os
import joblib
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings
warnings.filterwarnings("ignore")

class GenderPredictorHybrid:
    def __init__(self, model_dir = "decission_model\\gender_model_hybrid"):
        """
        Load hybrid model components: classifier, label encoder, SBERT, TF-IDF
        """
        self.model_dir = model_dir
        try:
            self.clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
            self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
            self.tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
            self.sbert_model = SentenceTransformer(os.path.join(model_dir, "sbert_model"))
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model components: {e}")

    def predict(self, name):
        """
        Predict gender using hybrid embedding (SBERT + TF-IDF)
        Returns: (predicted_label, confidence_percentage)
        """
        if not isinstance(name, str) or not name.strip():
            return ("Invalid input", 0.0)

        try:
            name_clean = name.strip()

            # Encode using SBERT
            sbert_emb = self.sbert_model.encode([name_clean])

            # Encode using TF-IDF
            tfidf_emb = self.tfidf.transform([name_clean]).toarray()

            # Combine embeddings
            combined_emb = np.hstack([sbert_emb, tfidf_emb])

            # Predict label and probabilities
            pred = self.clf.predict(combined_emb)[0]
            pred_proba = self.clf.predict_proba(combined_emb)[0]

            label = self.label_encoder.inverse_transform([pred])[0]
            confidence = round(100 * pred_proba[pred], 2)  # Convert to percentage

            return label, confidence
        except Exception as e:
            return (f"Prediction error: {str(e)}", 0.0)