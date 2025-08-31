#!/usr/bin/env python3
"""
OnTheSnow scraper for Australian ski resort information and reviews
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
warnings.filterwarnings('ignore')

# List of Australian ski resorts to scrape
AUSTRALIAN_RESORTS = [
    "https://www.onthesnow.com/australia/thredbo-alpine-resort/ski-resort",
    "https://www.onthesnow.com/australia/perisher/ski-resort",
    "https://www.onthesnow.com/australia/mt-buller/ski-resort",
    "https://www.onthesnow.com/australia/falls-creek-alpine-resort/ski-resort",
    "https://www.onthesnow.com/australia/mt-hotham/ski-resort",
    "https://www.onthesnow.com/australia/mt-baw-baw-alpine-resort/ski-resort",
    "https://www.onthesnow.com/australia/charlotte-pass/ski-resort",
    "https://www.onthesnow.com/australia/selwyn-snowfields/ski-resort"
]

# Chrome options for Selenium
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--window-size=1920,1080')
chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

def scrape_resort_overview(resort_url):
    """Scrape resort overview data from OnTheSnow"""
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(resort_url)
        wait = WebDriverWait(driver, 10)
        main_content = wait.until(
            EC.presence_of_element_located((By.ID, "ski-resort_main_content__Njkuw"))
        )
        time.sleep(3)
        page_source = driver.page_source
        driver.quit()
        soup = BeautifulSoup(page_source, 'html.parser')
        resort_data = extract_resort_data(soup, resort_url)
        return resort_data
    except Exception as e:
        print(f"Error scraping resort {resort_url}: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

def extract_resort_data(soup, resort_url):
    """Extract resort data from BeautifulSoup object"""
    resort_data = {
        'url': resort_url,
        'scraped_at': datetime.now().isoformat(),
        'data_source': 'OnTheSnow',
        'resort_name': '',
        'resort_key': resort_url.split('/')[-2]
    }
    
    # Extract resort name
    title_elem = soup.find('h1')
    if title_elem:
        resort_data['resort_name'] = title_elem.get_text(strip=True)
    
    # Extract main content sections
    main_content = soup.find('section', id="ski-resort_main_content__Njkuw")
    if main_content:
        extract_all_sections(main_content, resort_data)
        extract_specific_data(main_content, resort_data)
    
    return resort_data

def extract_all_sections(main_content, resort_data):
    """Extract all sections from main content"""
    sections = main_content.find_all('section')
    
    for section in sections:
        section_id = section.get('id', '')
        if not section_id:
            continue
            
        title = extract_section_title(section)
        content = extract_section_content(section)
        
        if title and content:
            section_key = f"section_{title.lower().replace(' ', '_')}"
            resort_data[f"{section_key}_title"] = title
            resort_data[f"{section_key}_content"] = content

def extract_section_title(section):
    """Extract section title"""
    title_elem = section.find(['h2', 'h3'])
    if title_elem:
        return title_elem.get_text(strip=True)
    return None

def extract_section_content(section):
    """Extract section content"""
    # Try to find content in description divs first
    desc_divs = section.find_all('div', class_='styles_section_description__hBWL0')
    if desc_divs:
        content_parts = []
        for div in desc_divs:
            text = div.get_text(strip=True, separator=' ')
            if len(text) > 50:
                content_parts.append(text)
        if content_parts:
            return ' '.join(content_parts)
    
    # Fallback to direct paragraph content
    paragraphs = section.find_all('p')
    content_parts = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if len(text) > 30:
            content_parts.append(text)
    
    return ' '.join(content_parts) if content_parts else ''

def extract_specific_data(main_content, resort_data):
    """Extract specific resort data like terrain, lifts, etc."""
    # Extract terrain breakdown
    terrain_widget = main_content.find('article', id="terrain_overview_widget")
    if terrain_widget:
        extract_terrain_breakdown(terrain_widget, resort_data)
    
    # Extract lifts breakdown
    lifts_widget = main_content.find('div', id="total_lifts_widget")
    if lifts_widget:
        extract_lifts_breakdown(lifts_widget, resort_data)
    
    # Extract important dates
    dates_section = main_content.find('section', id="styles_whenToGo__Izgmj")
    if dates_section:
        extract_important_dates(dates_section, resort_data)
    
    # Extract main description
    main_desc_section = main_content.find('section', id="styles_slider__Duvzd")
    if main_desc_section:
        desc_div = main_desc_section.find('div', class_='styles_section_description__hBWL0')
        if desc_div:
            main_desc = desc_div.get_text(strip=True, separator=' ')
            if len(main_desc) > 100:
                resort_data['main_description_title'] = 'Resort Overview'
                resort_data['main_description_content'] = main_desc

def extract_terrain_breakdown(terrain_widget, resort_data):
    """Extract terrain breakdown data"""
    boxes = terrain_widget.find_all('div', class_='styles_box__cUvBW')
    for box in boxes:
        title_elem = box.find('div', class_='styles_title__zz3Sm')
        metric_elem = box.find('div', class_='styles_metric__z_U_F')
        if title_elem and metric_elem:
            title = title_elem.get_text(strip=True).lower().replace(' ', '_')
            metric = metric_elem.get_text(strip=True)
            resort_data[title] = metric

def extract_lifts_breakdown(lifts_widget, resort_data):
    """Extract lifts breakdown data"""
    boxes = lifts_widget.find_all('div', class_='styles_box__cUvBW')
    for box in boxes:
        title_elem = box.find('div', class_='styles_title__zz3Sm')
        metric_elem = box.find('div', class_='styles_metric__z_U_F')
        if title_elem and metric_elem:
            title = title_elem.get_text(strip=True).lower().replace(' ', '_')
            metric = metric_elem.get_text(strip=True)
            resort_data[title] = metric

def extract_important_dates(dates_section, resort_data):
    """Extract important dates data"""
    date_divs = dates_section.find_all('div')
    for div in date_divs:
        title_elem = div.find('h4')
        value_elem = div.find('p')
        if title_elem and value_elem:
            title = title_elem.get_text(strip=True).lower().replace(' ', '_')
            value = value_elem.get_text(strip=True)
            resort_data[title] = value

def scrape_resort_reviews(resort_url):
    """Scrape reviews from the resort reviews page"""
    reviews_url = resort_url.replace('/ski-resort', '/reviews')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(reviews_url)
        wait = WebDriverWait(driver, 15)
        
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(5)
        
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='review'], article, section[class*='review']")))
        except:
            pass
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        see_more_clicked = 0
        max_clicks = 15
        
        while see_more_clicked < max_clicks:
            try:
                see_more_selectors = [
                    "//button[contains(text(), 'See More')]",
                    "//button[contains(text(), 'Load More')]",
                    "//button[contains(text(), 'Show More')]",
                    "//a[contains(text(), 'See More')]",
                    "//div[contains(text(), 'See More')]",
                    "//*[contains(@class, 'loadMore') or contains(@class, 'see-more') or contains(@class, 'load-more')]"
                ]
                
                button_found = False
                for selector in see_more_selectors:
                    try:
                        see_more_button = driver.find_element(By.XPATH, selector)
                        if see_more_button.is_displayed() and see_more_button.is_enabled():
                            driver.execute_script("arguments[0].scrollIntoView();", see_more_button)
                            time.sleep(1)
                            see_more_button.click()
                            see_more_clicked += 1
                            time.sleep(3)
                            button_found = True
                            break
                    except:
                        continue
                
                if not button_found:
                    break
                    
            except Exception as e:
                break
        
        time.sleep(5)
        page_source = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(page_source, 'html.parser')
        reviews = extract_reviews_from_page(soup)
        
        return reviews
        
    except Exception as e:
        print(f"Error scraping reviews from {reviews_url}: {e}")
        if 'driver' in locals():
            try:
                driver.quit()
            except:
                pass
        return []

def extract_reviews_from_page(soup):
    """Extract reviews from the reviews page HTML"""
    reviews = []
    
    review_containers = soup.find_all(['div', 'article'], class_=re.compile(r'review|comment|rating', re.I))
    user_content = soup.find_all(['div', 'p'], string=re.compile(r'(ski|snow|terrain|lift|resort)', re.I))
    rating_elements = soup.find_all(class_=re.compile(r'star|rating|score', re.I))
    review_sections = soup.find_all(['section', 'div'], attrs={'data-testid': re.compile(r'review', re.I)})
    if not review_sections:
        review_sections = soup.find_all(['div'], class_=re.compile(r'styles_wrapper|styles_review', re.I))
    
    text_elements = soup.find_all(['p', 'div'], string=lambda text: text and len(text.strip()) > 100)
    substantial_text = [elem for elem in text_elements if any(keyword in elem.get_text().lower() 
                  for keyword in ['ski', 'snow', 'resort', 'mountain', 'lift', 'terrain', 'powder'])]
    
    all_containers = review_containers + review_sections
    
    if all_containers:
        for i, container in enumerate(all_containers[:10]):
            review_data = extract_single_review(container)
            if review_data and any(review_data.values()):
                reviews.append(review_data)
    
    if not reviews and substantial_text:
        for i, elem in enumerate(substantial_text[:5]):
            text_content = elem.get_text(strip=True)
            if len(text_content) > 50:
                review_data = {
                    'reviewer_name': f"User {i+1}",
                    'review_text': text_content[:500],
                    'pros': '',
                    'cons': '',
                    'rating': None
                }
                reviews.append(review_data)
    
    if reviews:
        seen_texts = set()
        unique_reviews = []
        for review in reviews:
            review_text = review.get('review_text', '')
            if review_text and review_text not in seen_texts:
                seen_texts.add(review_text)
                unique_reviews.append(review)
        reviews = unique_reviews
    
    return reviews

def extract_single_review(container):
    """Extract a single review from a container element"""
    try:
        review_data = {
            'reviewer_name': '',
            'review_text': '',
            'pros': '',
            'cons': '',
            'rating': None
        }
        
        container_text = container.get_text(strip=True, separator=' ')
        
        if len(container_text) < 30:
            return None
            
        if any(skip_text in container_text.lower() for skip_text in [
            'order by', 'most recent', 'highest rating', 'lowest rating',
            'all reviews', 'total reviews', 'star reviews',
            'beginnerintermediate', 'expertall terrain',
            'family friendlyapres ski', 'terrain parkoverall value'
        ]):
            return None
        
        text_elements = container.find_all(string=True)
        potential_names = []
        
        for text in text_elements:
            text = text.strip()
            if text and len(text) > 1 and len(text) < 30:
                if (text[0].isupper() and 
                    not any(word in text.lower() for word in [
                        'review', 'rating', 'ski', 'resort', 'terrain', 'snow',
                        'mountain', 'lift', 'trail', 'beginner', 'intermediate',
                        'expert', 'family', 'friendly', 'apres', 'park', 'overall',
                        'value', 'total', 'star', 'most', 'recent', 'highest',
                        'lowest', 'order', 'by', 'all'
                    ]) and
                    not text.isdigit() and
                    ' ' not in text
                ):
                    potential_names.append(text)
        
        name_patterns = [
            r'([A-Z][a-z]+)(?:\s+said|:|\s+wrote|\s+reviewed)',
            r'^([A-Z][a-z]+)\s+',
            r'([A-Z][a-z]+)(?=\s*\w+\s+is\s+)',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, container_text)
            for match in matches:
                if len(match) > 1 and len(match) < 20:
                    potential_names.append(match)
        
        if potential_names:
            review_data['reviewer_name'] = potential_names[0]
        else:
            review_data['reviewer_name'] = 'Anonymous'
        
        paragraphs = container.find_all(['p', 'div'], string=lambda text: text and len(text.strip()) > 20)
        
        review_text_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if any(skip in text.lower() for skip in [
                'star reviews', 'total reviews', 'order by', 'most recent',
                'family friendly', 'terrain park', 'overall value',
                'beginner', 'intermediate', 'expert', 'all terrain'
            ]):
                continue
                
            if any(keyword in text.lower() for keyword in [
                'ski', 'snow', 'resort', 'mountain', 'lift', 'terrain',
                'powder', 'trail', 'slope', 'run', 'beautiful', 'expensive',
                'great', 'good', 'bad', 'love', 'hate', 'enjoy', 'fun'
            ]):
                review_text_parts.append(text)
        
        if not review_text_parts:
            sentences = re.split(r'[.!?]\s+', container_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 30 and 
                    any(keyword in sentence.lower() for keyword in [
                        'ski', 'snow', 'resort', 'mountain', 'lift', 'terrain',
                        'powder', 'trail', 'slope', 'run', 'beautiful', 'expensive',
                        'great', 'good', 'bad', 'love', 'hate', 'enjoy', 'fun'
                    ])):
                    review_text_parts.append(sentence)
        
        if review_text_parts:
            review_data['review_text'] = ' '.join(review_text_parts[:3])
        else:
            review_data['review_text'] = container_text[:300]
        
        rating_text = container.find(string=re.compile(r'\d+\.\d+|\d+/\d+|\d+\s*star'))
        if rating_text:
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                try:
                    review_data['rating'] = float(rating_match.group(1))
                except ValueError:
                    pass
        
        if (review_data['review_text'] and 
            len(review_data['review_text'].strip()) > 20 and
            review_data['reviewer_name']):
            return review_data
        
        return None
        
    except Exception as e:
        return None

def scrape_all_australian_resorts():
    """Scrape all Australian ski resorts"""
    all_resort_data = {}
    
    for resort_url in AUSTRALIAN_RESORTS:
        resort_key = resort_url.split('/')[-2]
        resort_data = scrape_resort_overview(resort_url)
        
        if resort_data:
            all_resort_data[resort_key] = resort_data
        
        time.sleep(2)
    
    return all_resort_data

def scrape_all_australian_resort_reviews():
    """Scrape reviews from all Australian ski resorts"""
    all_reviews_data = {}
    
    for resort_url in AUSTRALIAN_RESORTS:
        resort_key = resort_url.split('/')[-2]
        reviews_data = scrape_resort_reviews(resort_url)
        
        if reviews_data:
            all_reviews_data[resort_key] = reviews_data
        
        time.sleep(3)
    
    return all_reviews_data

def save_resort_data(resort_data, output_file="onthesnow_resort_data.csv"):
    """Save resort data to CSV file"""
    try:
        flattened_data = []
        for resort_key, data in resort_data.items():
            row = {'resort_key': resort_key}
            row.update(data)
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(output_file, index=False)
        print(f"Resort data saved to: {output_file}")
        return df
    except Exception as e:
        print(f"Error saving resort data: {e}")
        return None

def save_reviews_data(reviews_data, filename='onthesnow_reviews_data.csv'):
    """Save reviews data to CSV file"""
    try:
        flattened_reviews = []
        for resort_key, reviews in reviews_data.items():
            for review in reviews:
                review_row = {
                    'resort_key': resort_key,
                    'reviewer_name': review.get('reviewer_name', ''),
                    'review_text': review.get('review_text', ''),
                    'rating': review.get('rating'),
                    'date': review.get('date'),
                    'pros': review.get('pros', ''),
                    'cons': review.get('cons', ''),
                    'scraped_at': datetime.now().isoformat()
                }
                flattened_reviews.append(review_row)
        
        df = pd.DataFrame(flattened_reviews)
        df.to_csv(filename, index=False)
        print(f"Reviews data saved to: {filename}")
        return df
    except Exception as e:
        print(f"Error saving reviews data: {e}")
        return None

def main():
    """Main function to run the OnTheSnow scraper"""
    print("Starting OnTheSnow Australian Ski Resorts scraping...")
    
    resort_data = scrape_all_australian_resorts()
    reviews_data = scrape_all_australian_resort_reviews()
    
    save_resort_data(resort_data, 'onthesnow_resorts_data.csv')
    save_reviews_data(reviews_data, 'onthesnow_reviews_data.csv')
    
    print("OnTheSnow scraping completed successfully!")

if __name__ == "__main__":
    main() 