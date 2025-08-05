import os
import re

class ResumeNameExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    def clean_filename(self):
        # Remove file extension
        base_name = os.path.splitext(self.file_name)[0]

        # Replace underscores and hyphens with space
        base_name = re.sub(r'[_\-]', ' ', base_name)

        # Normalize to lowercase for cleanup
        base_name_lower = base_name.lower()

        # Remove unwanted keywords (case insensitive)
        keywords = ['resume', 'cv', 'profile', 'final', 'updated', 'new', 'design', 'designed']
        for word in keywords:
            base_name_lower = re.sub(rf'\b{word}\b', '', base_name_lower)

        # Remove digits and extra non-alphabet characters
        base_name_clean = re.sub(r'[^a-z\s]', '', base_name_lower)
        base_name_clean = re.sub(r'\s+', ' ', base_name_clean).strip()

        # Capitalize properly
        return base_name_clean.title()

    def extract_name(self):
        return self.clean_filename()