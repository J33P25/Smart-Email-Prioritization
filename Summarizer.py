import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from Preprocessing import Preprocessing  

OUTPUT_FILE = "final_analytics_data.csv"
MODEL_NAME  = "facebook/bart-large-cnn"
BATCH_SIZE  = 4  

class Stage4Summarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Stage 4] Initializing BART on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def summarize(self, text):
        if not isinstance(text, str) or len(text.split()) < 12:
            return text 

        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":

    print("\n[Stage 2] Running Preprocessing...")
    p = Preprocessing(sample_size=1000)
    p.apply_parse()
    p.apply_cleaning()
  
    df = p.df


    s4 = Stage4Summarizer()
    print(f"\n[Stage 4] Summarizing {len(df)} emails...")
    
    summaries = []
    texts = df["clean_body_summary"].fillna("").tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BART Processing"):
        batch = texts[i : i + BATCH_SIZE]
        for email_text in batch:
            summaries.append(s4.summarize(email_text))

    df["bart_summary"] = summaries

    final_cols = ["date", "from", "subject", "bart_summary", "clean_body_summary"]
    df[final_cols].to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[Success] Stage 4 Complete. Final CSV saved: {OUTPUT_FILE}")