import re

class Cleaner:

    def remove_replies(self, text):
        text = re.split(r'-----Original Message-----|On .* wrote:', text)[0]
        text = re.sub(r'>.*', '', text)
        return text

    def remove_signature(self, text):
        lines = text.split("\n")
        return "\n".join(lines[:-3])
    
    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_special_chars(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def clean(self, text):
        if not isinstance(text, str):
            return ""

        text = self.remove_replies(text)
        text = self.remove_signature(text)
        text = self.normalize_text(text)
        text = self.remove_special_chars(text)

        return text