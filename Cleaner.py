import re

class Cleaner:

    def remove_replies(self, text):
        text = re.split(r'-----Original Message-----|On .* wrote:|(?:-{5,}\s*Forwarded by)', text)[0]
        text = re.sub(r'>.*', '', text, flags=re.MULTILINE)
        return text

    def remove_signature(self, text):
        lines = text.split("\n")
        # Scan bottom-up for real signature markers instead of blindly cutting 3 lines
        # (blind cut empties short emails and silently removes real content)
        sig_markers = re.compile(
            r'^(--|best regards|regards|thanks|thank you|sincerely|cheers|'
            r'sent from my|get your (private|free).*(email|e-mail)|'
            r'share information about yourself)',
            re.IGNORECASE
        )
        cutoff = len(lines)
        for i in range(len(lines) - 1, max(len(lines) - 15, -1), -1):
            if sig_markers.search(lines[i].strip()):
                cutoff = i
                break
        return "\n".join(lines[:cutoff])
    
    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_special_chars(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def remove_urls(self, text):
        # Remove http/https/www URLs (were leaking into tokens)
        return re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    
    def is_usable(self, text):
        # Reject empty, near-empty, or system/boilerplate emails
        if not isinstance(text, str) or len(text.split()) < 5:
            return False
        boilerplate = re.search(
            r'(immediate action required|do not delete|'
            r'please click on the following link|unsubscribe|'
            r'unique id.*participant)',
            text, re.IGNORECASE
        )
        return not boilerplate

    def clean_for_summarization(self, text): # minimal cleaning for summarization
        if not isinstance(text, str):
            return ""
        text = self.remove_replies(text)
        text = self.remove_signature(text)
        text = self.remove_urls(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n+', ' ', text)
        return text if self.is_usable(text) else ""


    def clean_for_classification(self, text): # more aggressive cleaning for classification
        text = self.clean_for_summarization(text)
        if not text:
            return ""
        text = text.lower()
        text = self.remove_special_chars(text)
        return text