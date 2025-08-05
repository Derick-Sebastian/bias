from sentence_transformers import SentenceTransformer, util
import warnings

# Suppress warnings from transformers, etc.
warnings.filterwarnings("ignore")

class ResumeComparator:
    def __init__(self, ideal_resume_path = "decission_model\\samples\\IDEAL RES.txt"
):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ideal_resume = self._load_resume(ideal_resume_path)
        self.ideal_embedding = self.model.encode(self.ideal_resume, convert_to_tensor=True)

    def _load_resume(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def compare_with_file(self, sample_resume_path):
        sample_resume = self._load_resume(sample_resume_path)
        return self._compare(sample_resume)

    def compare_with_text(self, sample_resume_text):
        return self._compare(sample_resume_text)

    def _compare(self, sample_text):
        sample_embedding = self.model.encode(sample_text, convert_to_tensor=True)
        similarity = util.cos_sim(self.ideal_embedding, sample_embedding).item()
        return round(similarity * 100, 2)
