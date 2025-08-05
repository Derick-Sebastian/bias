import os
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text


class ResumeConverter:
    def __init__(self, input_path):
        self.input_path = input_path
        self.ext = os.path.splitext(input_path)[-1].lower()
        self.supported_extensions = [".pdf", ".docx", ".txt", ".rtf", ".html", ".htm", ".tex", ".latex"]

        if self.ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {self.ext}. Supported formats are: {', '.join(self.supported_extensions)}")

    def extract_text(self):
        if self.ext == ".pdf":
            return self._extract_text_from_pdf()
        elif self.ext == ".docx":
            return self._extract_text_from_docx()
        elif self.ext == ".txt":
            return self._extract_text_from_txt()
        elif self.ext == ".rtf":
            return self._extract_text_from_rtf()
        elif self.ext in [".html", ".htm"]:
            return self._extract_text_from_html()
        elif self.ext in [".tex", ".latex"]:
            return self._extract_text_from_tex()

    def convert_to_txt(self):
        text = self.extract_text()
        print(f"[✔] Extracted text from '{self.input_path}' without saving to file.")
        return text

    def _extract_text_from_pdf(self):
        text = ""
        with pdfplumber.open(self.input_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _extract_text_from_docx(self):
        doc = Document(self.input_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _extract_text_from_txt(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_text_from_rtf(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content)

    def _extract_text_from_html(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        return soup.get_text(separator="\n")

    def _extract_text_from_tex(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        content = []
        for line in lines:
            if not line.strip().startswith("\\"):
                content.append(line)
        return "".join(content)
