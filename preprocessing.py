import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from email import message_from_string
from Cleaner import Cleaner
import kagglehub


nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

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

    def view_email(self):
        print(self.df.head())

    def apply_cleaning(self):
        self.df["clean_body_summary"] = self.df["body"].apply(
            self._cleaner.clean_for_summarization
        )
        self.df["clean_body_classify"] = self.df["body"].apply(
            self._cleaner.clean_for_classification
        )

    def tokenization(self):

        self.df['Tokens'] = self.df['clean_body'].apply(lambda x: word_tokenize(x))

if __name__ == "__main__":
    p = Preprocessing(sample_size=1000)
    p.apply_parse()
    p.view_email()
    p.apply_cleaning()