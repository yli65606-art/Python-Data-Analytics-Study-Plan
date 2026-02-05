import sys
import os

# This tells Python to look in the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# NOW you can import your modules

import news_scraper
import data_cleaner
import datetime

def main():
    print(f"--- Pipeline Started: {datetime.datetime.now()} ---")
    
    # Step 1: Extract
    news_scraper.run_scrape()
    
    # Step 2: Transform/Clean
    data_cleaner.run_clean()
    
    print(f"--- Pipeline Finished Successfully ---\n")

if __name__ == "__main__":
    main()