"""
SELENIUM SCRAPING TWITTER.
"""
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common import exceptions
import time
import pandas as pd
import urllib
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support import expected_conditions as EC
from helpers import clean_text,merge_file,clean_file, build_dataset
from model import build_model


# Variable Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
#service = ChromeService(executable_path="./drivers/chromedriver")
service = ChromeService(executable_path=ChromeDriverManager().install())
browser = webdriver.Chrome(service=service, options=chrome_options)
browser.set_page_load_timeout(10)
base_url = 'https://twitter.com/search?q='
# number of tweets for each words by default
number_tweets = 500
# default words ! you chan change it
words = ['content','raison', 'nouvelle', 'bravo', 'felicitations','aide','assistance','plaisir','confort',\
         'heureux', 'super','enthousiame','epanouissement','amour','cherir',\
         'adorer','faveur','foi', 'pute' , 'salope', 'pd', 'connasse', 'encule''attaque','traque',\
         'hante','ronge','bite', 'cul', 'connard', 'batard', 'enfoire', 'abruti gueule',\
         'harcele', 'vilipende','chatie','blesse','bitch','nègre','salaud','sexe','putain','levrette',\
         'homophobie','islam','nègro','imbécile','exclave','lgbt','gana'
         ]


def get_tweets(keywords= []):
    """
    get tweets by each word
    :param keywords: list of words
    """

    if not keywords:
        keywords = words

    for i, query in enumerate(keywords):
        tweets = []
        if query is not None and isinstance(query, str):
            query_encode = urllib.parse.quote(query) + urllib.parse.quote(' lang:fr')
            url = f"{base_url}{query_encode}&src=typed_query&f=live"

        try:
            # scroll management
            scrolling = True
            browser.get(url)
            while scrolling:
                if WebDriverWait(browser, 40).until(
                        EC.presence_of_element_located((By.XPATH, '//div[@data-testid="cellInnerDiv"]'))):

                    e = browser.find_element(By.XPATH, '//article[@data-testid="tweet"]')

                    posts = get_tweet_data(e)
                    # no empty posts and be manage tweets duplications
                    if posts:
                        [tweets.append(tweet) for tweet in posts if tweet is not None and tweet not in tweets]
                    # save these tweets in dataframe
                    pd.DataFrame(data={'text': tweets}).to_csv(f'./data/{query}.csv', index=False)

                # attempt
                scroll_attempt = 0
                try:
                    last_position = browser.execute_script("return document.body.scrollHeight;")
                except exceptions.WebDriverException as e:
                    print(e.msg)
                    print('Une erreur vient de se produire : Note 1')
                    browser.close()
                    scrolling = False
                while True:
                    # save data
                    if len(tweets) >= number_tweets:
                        scrolling = False
                        print('Taille atteinte pour :' + query)
                        # browser.close()
                        break

                    try:
                        # check scroll position
                        browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                        time.sleep(3)
                    except exceptions.WebDriverException as e:
                        print(e.msg)
                        scrolling = False
                        break

                    try:
                        curr_position = browser.execute_script("return document.body.scrollHeight ;")
                    except exceptions.WebDriverException as E:
                        print(E.msg)
                        print('Scrolling stopped.. break and close browser')
                        scrolling = False
                        break
                    if last_position == curr_position:
                        scroll_attempt += 1

                        if scroll_attempt >= 2:
                            scrolling = False
                            print('End scrolling .. go out..and close browser ')
                            break
                        else:
                            time.sleep(2)
                    else:
                        last_position = curr_position
                        # browser.quit()
                        print(f'Scrolling ...: {query}')
                        # scrolling = False
                        break
            # browser.close()
        except StaleElementReferenceException as Exception:
            print(Exception.msg)
            browser.quit()

    # all ok close the browser
    browser.close()


def get_tweet_data(card):
    """
    :param card:
    :return: list
    """
    try:
        if card is not None:
            el_texts = card.find_elements(by=By.XPATH, value='//div[@data-testid="tweetText"]')
            posts = [clean_text(tweet.text.strip()) for tweet in el_texts if el_texts is not None  and clean_text(tweet.text.strip()) ]
    except exceptions as e:
        print(e.msg)
        return

    return posts


def initialize():
    """
    initialise by changing if you want the keywords
    :return: list
    """
    print('*****' * 20)
    print(f'\tDefault words :{words}')
    print('*****' * 20)
    res = input('Would you like to change this words (yes/no) :')
    if res == 'yes' or 'y':
        keywords = []
        again = True
        while again:
            keywords.append(input('keyword :'))
            res = input('Continue (yes/no):')
            if res.lower() == 'no' or res in 'no':
                again = False
                break
        print(keywords)
    else:
        keywords = words
    print('*****'*20)
    return keywords


if __name__ == '__main__':
    # get keywords
    keywords = initialize()
    # get tweets containing this keywords
    get_tweets(keywords)
    # merge all csv files into one.
    merge_file()
    # cleaning dataset
    clean_file()
    # build dataset # labeliser
    print('Build dataset and annotate')
    build_dataset()
    print('Create Model')
    # build model
    build_model()
