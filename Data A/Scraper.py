import requests
import time
import csv
from bs4 import BeautifulSoup

# 1. Setup the CSV file
# We open the file 'quotes.csv' in write mode ('w')
with open('quotes_results.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Quote', 'Author', 'Tags'])

    # 2. Setup Scraping Logic
    base_url = 'http://quotes.toscrape.com'
    current_url = '/page/1/'

    while current_url:
        print(f"Scraping: {base_url + current_url}...")
        response = requests.get(base_url + current_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            quotes = soup.find_all('div', class_='quote')

            for quote in quotes:
                text = quote.find('span', class_='text').text
                author = quote.find('small', class_='author').text
                tags_list = [tag.text for tag in quote.find_all('a', class_='tag')]
                
                # Convert list of tags into a single string separated by commas
                tags_string = ", ".join(tags_list)

                # 3. Write data to the CSV file
                writer.writerow([text, author, tags_string])
            
            # Find the "Next" button
            next_button = soup.find('li', class_='next')
            if next_button:
                current_url = next_button.find('a')['href']
                time.sleep(1) # Polite delay
            else:
                current_url = None
                print("\nDone! Your data is saved in 'quotes_results.csv'.")
        else:
            print("Error reaching the site.")
            break