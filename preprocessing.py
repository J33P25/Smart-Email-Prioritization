import kagglehub
import pandas as pd
from Cleaner import Cleaner
from email import message_from_string
import os
import re

PATH = kagglehub.dataset_download('wcukierski/enron-email-dataset')

class Preprocessing:
    def __init__(self):
        self.file_path = os.path.join(PATH,"emails.csv")
        self.df = pd.read_csv(self.file_path)
        self.df = self.df.head(1000)

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
        cleaner = Cleaner()
        self.df["clean_body"] = self.df["body"].apply(cleaner.clean)
        print(self.df["clean_body"][1])

p = Preprocessing()
p.apply_parse()
p.view_email()
p.apply_cleaning()