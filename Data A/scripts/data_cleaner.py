import pandas as pd
import os
from datetime import datetime

def run_clean():
    raw_path = 'data/raw_news.csv'
    master_path = 'data/master_news_archive.csv'
    
    if not os.path.exists(raw_path):
        return

    df = pd.read_csv(raw_path)
    
    # 1. Add Metadata
    df['Scrape_Date'] = datetime.now().strftime("%Y-%m-%d")
    
    # 2. Text Normalization (Crucial for multi-source mining)
    # Convert everything to lowercase and remove leading/trailing whitespace
    df['Headline'] = df['Headline'].str.lower().str.strip()
    
    # 3. Deduplicate 
    # We remove duplicates based on the headline text
    df.drop_duplicates(subset=['Headline'], inplace=True)
    
    # 4. Filter Noise
    df = df[df['Headline'].str.split().str.len() > 4]
    
    # Append to master archive
    df.to_csv(master_path, mode='a', index=False, header=not os.path.exists(master_path))
    print(f"Cleaned and merged data from {df['Source'].nunique()} sources.")