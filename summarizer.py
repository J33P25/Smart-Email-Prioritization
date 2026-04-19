import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_FILE  = "preprocessed_emails.pkl"
OUTPUT_FILE = "final_analytics_data.csv"
MODEL_NAME  = "facebook/bart-large-cnn"
BATCH_SIZE  = 4  # Reduced slightly for 4GB VRAM safety on RTX 3050

class Summarizer:
    def __init__(self):
        # 1. Device Setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        status = f"GPU ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU"
        print(f"[Summarizer] Initializing {MODEL_NAME} on {status}...")

        # 2. Direct Model Loading (Bypasses deprecated pipeline tasks)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self.device)
        print("[Summarizer] Model and Tokenizer loaded successfully.")

    def _generate_summary(self, text):
        """Processes single text using the manual generate method."""
        if not isinstance(text, str) or len(text.split()) < 10:
            return text 

        # Tokenize and move to GPU
        inputs = self.tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # Generate sequence IDs
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode back to natural text
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def apply_summarization(self, df):
        print(f"[Summarizer] Processing {len(df)} emails...")
        texts = df["clean_body_summary"].fillna("").tolist()
        summaries = []

        # Sequential processing within the loop for easier memory management
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Summarizing"):
            batch = texts[i : i + BATCH_SIZE]
            for email in batch:
                summaries.append(self._generate_summary(email))

        df["bart_summary"] = summaries
        return df

if __name__ == "__main__":
    # 1. Load data
    print(f"[IO] Loading {INPUT_FILE}...")
    try:
        df_loaded = pd.read_pickle(INPUT_FILE)
    except FileNotFoundError:
        print(f"[Error] {INPUT_FILE} not found!")
        exit()

    # 2. Run Summarization
    s = Summarizer()
    df_final = s.apply_summarization(df_loaded)

    # 3. Export for Tableau / Power BI
    # We keep 'tokens' and 'clean_body_summary' for full project traceability
    cols = ["from", "to", "subject", "date", "clean_body_summary", "bart_summary", "tokens"]
    valid_cols = [c for c in cols if c in df_final.columns]
    
    df_final[valid_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Success] Final data saved to {OUTPUT_FILE}")