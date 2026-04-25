import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from email import message_from_string
from Cleaner import Cleaner
import kagglehub


nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

pd.set_option('display.max_rows', None)        # show all rows
pd.set_option('display.max_colwidth', 250)    # show full text in cells
pd.set_option('display.width', None)          # auto-detect terminal width

class Preprocessing:
    def __init__(self,sample_size=1000):
        path = kagglehub.dataset_download('wcukierski/enron-email-dataset')
        file_path = os.path.join(path, "emails.csv")
 
        self.df = pd.read_csv(file_path)
        if sample_size is not None:
            self.df = self.df.head(sample_size)
 
        self._cleaner = Cleaner()

    def parse_email_message(self,message):
        try:
            email_msg = message_from_string(message)

            # Standard headers
            from_ = email_msg.get("From")
            to = email_msg.get("To")
            subject = email_msg.get("Subject")
            date = email_msg.get("Date")

            # Extract body
            body = ""

            if email_msg.is_multipart():
                for part in email_msg.walk():
                    content_type = part.get_content_type()
                    
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors="ignore")
                            break
            else:
                payload = email_msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore")

            return {
                "from": from_,
                "to": to,
                "subject": subject,
                "date": date,
                "body": body.strip()
            }
        except Exception:
            return {
                "from": None,
                "to": None,
                "subject": None,
                "date": None,
                "body": None
            }
        
    def apply_parse(self):
        parsed = self.df["message"].apply(self.parse_email_message)
        parsed_df = pd.DataFrame(parsed.tolist())

        self.df = pd.concat([self.df, parsed_df], axis=1)

    def view_email(self,index=0):
        print(self.df.iloc[index])

    def apply_cleaning(self):
        self.df["clean_body_summary"] = self.df["body"].apply(self._cleaner.clean_for_summarization)
        self.df["clean_body_classify"] = self.df["body"].apply(self._cleaner.clean_for_classification)
        
        before = len(self.df)
        self.df = self.df[self.df["clean_body_classify"].str.strip() != ""].reset_index(drop=True)
        print(f"[Cleaning] Dropped {before - len(self.df)} unusable rows.")

    def tokenization(self):
        stop_words = set(stopwords.words('english'))
 
        def _tokenize(text):
            if not isinstance(text, str) or not text.strip():
                return []
            tokens = word_tokenize(text)
            return [t for t in tokens if t not in stop_words]
 
        self.df["tokens"] = self.df["clean_body_classify"].apply(_tokenize)
 
        # Report any rows that tokenized to nothing (edge cases)
        empty_tokens = (self.df["tokens"].apply(len) == 0).sum()
        if empty_tokens:
            print(f"[Tokenization] Warning: {empty_tokens} rows produced 0 tokens.")

    def lemmatization(self):
        lemmatizer = WordNetLemmatizer()
        
        def _lemmatize(tokens):
            return [lemmatizer.lemmatize(t) for t in tokens]
        self.df["Lemmatized_Tokens"] = self.df["tokens"].apply(_lemmatize)

    def helper_sample_tokens(self):
        print(f"[Tokenization] Sample tokens from first email: {self.df['tokens'].iloc[0]}")
        print(f"[Lemmatization] Sample lemmatized tokens from first email: {self.df['Lemmatized_Tokens'].iloc[0]}")


if __name__ == "__main__":
    p = Preprocessing(sample_size=1000)
    p.apply_parse()
    p.view_email(1)
    p.apply_cleaning()
    p.tokenization()
    p.lemmatization()
    p.helper_sample_tokens()