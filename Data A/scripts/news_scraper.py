# World news scrapping

import requests
from bs4 import BeautifulSoup
import csv
import os

# --- Configuration ---
# We use a dictionary to define how to find headlines on each specific site
NEWS_CONFIG = {
    'BBC': {
        'url': 'https://www.bbc.com/news',
        'tag': 'h2'
    },
    'Reuters': {
        'url': 'https://www.reuters.com/world/',
        # We target the specific Heading testid AND common link classes
        'attrs': {'data-testid': ['Heading', 'Link', 'Title']} 
    },
    'TheGuardian': {
        'url': 'https://www.theguardian.com/international',
        'tag': 'h3'
    }
}

def run_scrape():
    # Professional Headers to mimic a real Chrome browser and avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    raw_path = 'data/raw_news.csv'
    
    # Ensure the 'data' directory exists before writing
    if not os.path.exists('data'): 
        os.makedirs('data')

    print("Starting Scrape...")
    
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Headline'])

        for site, config in NEWS_CONFIG.items():
            print(f"Accessing {site}...")
            try:
                # 15-second timeout prevents the pipeline from 'hanging' if a site is slow
                response = requests.get(config['url'], headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Specific logic for Reuters based on your developer view discovery
                    if site == 'Reuters':
                        headlines = soup.find_all(attrs=config['attrs'])
                        headlines += soup.find_all(['h3', 'h6'])
                    else:
                        headlines = soup.find_all(config['tag'])
                    
                    count = 0
                    unique_headlines = set() # Avoid saving the same headline twice

                    for h in headlines:
                        text = h.get_text(strip=True)
                        # Filter for actual headlines (ignore short UI text like 'Menu' or 'Search')
                        if len(text) > 30 and text not in unique_headlines:
                            writer.writerow([site, text])
                            unique_headlines.add(text)
                            count += 1
                            
                    print(f"--- Successfully found {count} headlines from {site}")
                else:
                    print(f"--- Failed to reach {site}. Status Code: {response.status_code}")
                    
            except Exception as e:
                print(f"--- Error occurred while scraping {site}: {e}")

    print(f"\nScrape complete. All data saved to {raw_path}")

if __name__ == "__main__":
    run_scrape()