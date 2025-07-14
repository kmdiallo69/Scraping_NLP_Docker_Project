#!/usr/bin/env python3
"""
ToxiGuard French Tweets - Web Scraping Module

This module handles the automated collection of French tweets using Selenium WebDriver.
It scrapes tweets based on specified keywords, processes the data, and prepares it 
for machine learning model training.

Author: ToxiGuard Team
License: MIT
"""

# Standard library imports
import time
import urllib.parse
from typing import List, Optional

# Third-party imports
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions
from webdriver_manager.chrome import ChromeDriverManager

# Local imports
from helpers import clean_text, merge_file, clean_file, build_dataset
from model import build_model


# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# Base URL for Twitter search
BASE_URL = 'https://twitter.com/search?q='

# Default number of tweets to collect per keyword
DEFAULT_TWEET_COUNT = 500

# Default French keywords for toxicity detection training
# Mix of positive and negative content to create a balanced dataset
DEFAULT_KEYWORDS = [
    # Positive/Neutral words
    'content', 'raison', 'nouvelle', 'bravo', 'félicitations', 'aide', 
    'assistance', 'plaisir', 'confort', 'heureux', 'super', 'enthousiasme',
    'épanouissement', 'amour', 'chérir', 'adorer', 'faveur', 'foi',
    
    # Negative/Toxic words (for training purposes)
    'pute', 'salope', 'pd', 'connasse', 'enculé', 'attaque', 'traque',
    'hanté', 'ronge', 'bite', 'cul', 'connard', 'bâtard', 'enfoiré', 
    'abruti', 'gueule', 'harcelé', 'vilipendé', 'châtié', 'blessé', 
    'bitch', 'nègre', 'salaud', 'sexe', 'putain', 'levrette',
    'homophobie', 'islam', 'négro', 'imbécile', 'esclave', 'lgbt'
]


# =============================================================================
# SELENIUM WEBDRIVER SETUP
# =============================================================================

def setup_chrome_driver() -> webdriver.Chrome:
    """
    Configure and initialize Chrome WebDriver with optimal settings for scraping.
    
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    print("🔧 Setting up Chrome WebDriver...")
    
    # Configure Chrome options for headless browsing and performance
    chrome_options = webdriver.ChromeOptions()
    
    # Essential options for stability and performance
    chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
    chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # Avoid detection
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    # Optional: Enable headless mode (uncomment for server deployment)
    # chrome_options.add_argument('--headless')
    
    # Create Chrome service using automatically managed ChromeDriver
    service = ChromeService(executable_path=ChromeDriverManager().install())
    
    # Initialize browser with configured options
    browser = webdriver.Chrome(service=service, options=chrome_options)
    browser.set_page_load_timeout(10)  # Set 10-second timeout for page loads
    
    print("✅ Chrome WebDriver successfully initialized")
    return browser


# =============================================================================
# TWEET SCRAPING FUNCTIONS
# =============================================================================

def build_search_url(keyword: str) -> str:
    """
    Construct a Twitter search URL for a given keyword in French.
    
    Args:
        keyword (str): The search term/keyword
        
    Returns:
        str: Complete Twitter search URL with language filter
    """
    # URL encode the keyword and add French language filter
    encoded_keyword = urllib.parse.quote(keyword)
    encoded_lang_filter = urllib.parse.quote(' lang:fr')
    
    # Construct the complete search URL
    search_url = f"{BASE_URL}{encoded_keyword}{encoded_lang_filter}&src=typed_query&f=live"
    
    return search_url


def extract_tweet_data(card_element) -> Optional[List[str]]:
    """
    Extract text content from a tweet card element.
    
    Args:
        card_element: Selenium WebElement representing a tweet card
        
    Returns:
        Optional[List[str]]: List of cleaned tweet texts, or None if extraction fails
    """
    try:
        # Find all tweet text elements within the card
        text_elements = card_element.find_elements(by=By.XPATH, value='//div[@data-testid="tweetText"]')
        
        if not text_elements:
            return None
            
        # Extract and clean text from each element
        tweets = []
        for element in text_elements:
            raw_text = element.text.strip()
            if raw_text:  # Only process non-empty tweets
                cleaned_text = clean_text(raw_text)
                if cleaned_text:  # Only add successfully cleaned tweets
                    tweets.append(cleaned_text)
        
        return tweets if tweets else None
        
    except Exception as e:
        print(f"⚠️ Error extracting tweet data: {e}")
        return None


def scrape_tweets_for_keyword(browser: webdriver.Chrome, keyword: str, 
                             target_count: int = DEFAULT_TWEET_COUNT) -> List[str]:
    """
    Scrape tweets for a specific keyword using the provided browser instance.
    
    Args:
        browser (webdriver.Chrome): Chrome WebDriver instance
        keyword (str): Keyword to search for
        target_count (int): Target number of tweets to collect
        
    Returns:
        List[str]: List of collected and cleaned tweets
    """
    print(f"🔍 Starting to scrape tweets for keyword: '{keyword}'")
    
    # Build search URL and navigate to it
    search_url = build_search_url(keyword)
    browser.get(search_url)
    
    tweets = []
    scroll_attempts = 0
    max_scroll_attempts = 3
    
    # Wait for initial page load
    try:
        WebDriverWait(browser, 40).until(
            EC.presence_of_element_located((By.XPATH, '//div[@data-testid="cellInnerDiv"]'))
        )
    except exceptions.TimeoutException:
        print(f"⚠️ Timeout waiting for page load for keyword: {keyword}")
        return tweets
    
    # Main scrolling and scraping loop
    while len(tweets) < target_count:
        try:
            # Find tweet cards on current page
            tweet_cards = browser.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
            
            # Extract data from each tweet card
            for card in tweet_cards:
                try:
                    extracted_tweets = extract_tweet_data(card)
                    if extracted_tweets:
                        # Add unique tweets only (avoid duplicates)
                        for tweet in extracted_tweets:
                            if tweet not in tweets:
                                tweets.append(tweet)
                                
                        # Save progress periodically
                        if len(tweets) % 50 == 0:
                            df = pd.DataFrame({'text': tweets})
                            df.to_csv(f'./data/{keyword}_temp.csv', index=False)
                            print(f"📝 Saved {len(tweets)} tweets for '{keyword}'")
                            
                except Exception as e:
                    print(f"⚠️ Error processing tweet card: {e}")
                    continue
            
            # Check if we've reached our target
            if len(tweets) >= target_count:
                print(f"🎯 Target reached: {len(tweets)} tweets collected for '{keyword}'")
                break
            
            # Scroll down to load more tweets
            print(f"📜 Scrolling to load more tweets... Current count: {len(tweets)}")
            last_position = browser.execute_script("return document.body.scrollHeight;")
            browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(3)  # Wait for new content to load
            
            # Check if new content was loaded
            current_position = browser.execute_script("return document.body.scrollHeight;")
            if last_position == current_position:
                scroll_attempts += 1
                print(f"⏳ No new content loaded. Attempt {scroll_attempts}/{max_scroll_attempts}")
                
                if scroll_attempts >= max_scroll_attempts:
                    print(f"🔚 No more content available for '{keyword}'. Stopping.")
                    break
                    
                time.sleep(2)  # Wait a bit longer before next attempt
            else:
                scroll_attempts = 0  # Reset counter if new content was loaded
                
        except Exception as e:
            print(f"❌ Error during scraping for '{keyword}': {e}")
            break
    
    # Save final results
    if tweets:
        df = pd.DataFrame({'text': tweets})
        df.to_csv(f'./data/{keyword}.csv', index=False)
        print(f"💾 Final save: {len(tweets)} tweets for '{keyword}'")
    
    return tweets


def scrape_multiple_keywords(keywords: List[str], tweets_per_keyword: int = DEFAULT_TWEET_COUNT) -> None:
    """
    Scrape tweets for multiple keywords sequentially.
    
    Args:
        keywords (List[str]): List of keywords to scrape
        tweets_per_keyword (int): Number of tweets to collect per keyword
    """
    print(f"🚀 Starting multi-keyword scraping for {len(keywords)} keywords")
    print(f"📊 Target: {tweets_per_keyword} tweets per keyword")
    
    # Initialize browser once for all keywords
    browser = setup_chrome_driver()
    
    try:
        total_tweets = 0
        successful_keywords = 0
        
        for i, keyword in enumerate(keywords, 1):
            print(f"\n{'='*60}")
            print(f"🔄 Processing keyword {i}/{len(keywords)}: '{keyword}'")
            print(f"{'='*60}")
            
            try:
                tweets = scrape_tweets_for_keyword(browser, keyword, tweets_per_keyword)
                
                if tweets:
                    total_tweets += len(tweets)
                    successful_keywords += 1
                    print(f"✅ Successfully collected {len(tweets)} tweets for '{keyword}'")
                else:
                    print(f"❌ No tweets collected for '{keyword}'")
                    
            except Exception as e:
                print(f"❌ Failed to scrape '{keyword}': {e}")
                continue
            
            # Add delay between keywords to be respectful to the platform
            if i < len(keywords):
                print("⏱️ Waiting 5 seconds before next keyword...")
                time.sleep(5)
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"📈 SCRAPING COMPLETE - SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"✅ Successful keywords: {successful_keywords}/{len(keywords)}")
        print(f"📊 Total tweets collected: {total_tweets}")
        print(f"📝 Average tweets per successful keyword: {total_tweets/successful_keywords if successful_keywords > 0 else 0:.1f}")
        
    finally:
        # Always close the browser, even if an error occurs
        print("🔚 Closing browser...")
        browser.quit()


# =============================================================================
# USER INTERACTION FUNCTIONS
# =============================================================================

def get_user_keywords() -> List[str]:
    """
    Interactive function to get keywords from user input.
    
    Returns:
        List[str]: List of user-provided keywords or default keywords
    """
    print("🎯 KEYWORD CONFIGURATION")
    print("="*50)
    print(f"📝 Default keywords ({len(DEFAULT_KEYWORDS)} total):")
    
    # Display default keywords in a formatted way
    for i, keyword in enumerate(DEFAULT_KEYWORDS, 1):
        print(f"  {i:2d}. {keyword}")
    
    print("="*50)
    
    # Ask user if they want to use custom keywords
    while True:
        choice = input("❓ Do you want to use custom keywords? (yes/no): ").lower().strip()
        if choice in ['yes', 'y', 'oui']:
            return get_custom_keywords()
        elif choice in ['no', 'n', 'non']:
            print("✅ Using default keywords")
            return DEFAULT_KEYWORDS
        else:
            print("⚠️ Please answer 'yes' or 'no'")


def get_custom_keywords() -> List[str]:
    """
    Get custom keywords from user input.
    
    Returns:
        List[str]: List of custom keywords
    """
    keywords = []
    print("\n📝 Enter your custom keywords (press Enter after each keyword)")
    print("💡 Type 'done' when finished, or 'quit' to use default keywords")
    
    while True:
        keyword = input(f"Keyword #{len(keywords)+1}: ").strip()
        
        if keyword.lower() == 'done':
            if keywords:
                break
            else:
                print("⚠️ Please enter at least one keyword before typing 'done'")
                continue
        elif keyword.lower() == 'quit':
            print("🔄 Switching to default keywords")
            return DEFAULT_KEYWORDS
        elif keyword:
            keywords.append(keyword)
            print(f"✅ Added: '{keyword}' ({len(keywords)} keywords total)")
        else:
            print("⚠️ Please enter a valid keyword")
    
    print(f"\n✅ Custom keywords configured: {len(keywords)} keywords")
    return keywords


def get_tweet_count() -> int:
    """
    Get the desired number of tweets per keyword from user.
    
    Returns:
        int: Number of tweets to collect per keyword
    """
    print(f"\n🎯 TWEET COUNT CONFIGURATION")
    print(f"📊 Default: {DEFAULT_TWEET_COUNT} tweets per keyword")
    
    while True:
        response = input(f"❓ Enter number of tweets per keyword (or press Enter for default): ").strip()
        
        if not response:  # User pressed Enter
            return DEFAULT_TWEET_COUNT
        
        try:
            count = int(response)
            if count > 0:
                return count
            else:
                print("⚠️ Please enter a positive number")
        except ValueError:
            print("⚠️ Please enter a valid number")


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def main():
    """
    Main function that orchestrates the entire scraping and model building process.
    """
    print("🛡️ ToxiGuard French Tweets - Data Collection & Model Training")
    print("="*70)
    print("🎯 This script will:")
    print("   1. Collect French tweets based on specified keywords")
    print("   2. Clean and process the collected data")
    print("   3. Automatically label the data using Detoxify")
    print("   4. Train a machine learning model for toxicity detection")
    print("="*70)
    
    try:
        # Step 1: Get configuration from user
        print("\n📋 STEP 1: CONFIGURATION")
        keywords = get_user_keywords()
        tweet_count = get_tweet_count()
        
        print(f"\n🔧 Configuration Summary:")
        print(f"   📝 Keywords: {len(keywords)} total")
        print(f"   📊 Tweets per keyword: {tweet_count}")
        print(f"   🎯 Total target tweets: {len(keywords) * tweet_count}")
        
        # Confirmation before starting
        confirm = input("\n❓ Start scraping with this configuration? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y', 'oui']:
            print("❌ Operation cancelled by user")
            return
        
        # Step 2: Scrape tweets
        print("\n📋 STEP 2: TWEET COLLECTION")
        scrape_multiple_keywords(keywords, tweet_count)
        
        # Step 3: Process and merge data
        print("\n📋 STEP 3: DATA PROCESSING")
        print("🔄 Merging all CSV files...")
        merge_file()
        print("✅ Files merged successfully")
        
        print("🧹 Cleaning merged dataset...")
        clean_file()
        print("✅ Dataset cleaned successfully")
        
        # Step 4: Label data
        print("\n📋 STEP 4: DATA LABELING")
        print("🤖 Labeling data using Detoxify model...")
        build_dataset()
        print("✅ Dataset labeled successfully")
        
        # Step 5: Train model
        print("\n📋 STEP 5: MODEL TRAINING")
        print("🧠 Training toxicity detection model...")
        build_model()
        print("✅ Model trained and saved successfully")
        
        # Success message
        print("\n🎉 PROCESS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("📁 Generated files:")
        print("   📊 ./data/dataset.csv - Labeled training dataset")
        print("   🤖 ./fastapi/model/finalized_tfidf.sav - Trained model")
        print("   🔤 ./fastapi/model/vectorizer_tfidf.sav - Text vectorizer")
        print("\n🚀 You can now start the API with: cd fastapi && python main.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation interrupted by user (Ctrl+C)")
        print("🔚 Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Please check the error message and try again")


if __name__ == '__main__':
    main()
