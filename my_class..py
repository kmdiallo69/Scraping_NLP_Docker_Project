
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

### import helpers

from helpers import clean_text,merge_file,clean_file,label, build_dataset
from model import build_model

class SeleniumClient(object):
    def __init__(self):

        # intialize parameters
        self.chrome_options = webdriver.ChromeOptions()
        #self.chrome_options.add_argument('headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        #self.service = ChromeService(executable_path=ChromeDriverManager().install())
        self.service = ChromeService(executable_path="./drivers/chromedriver")
        # you need to provide the path of chromedriver in your system
        self.browser = webdriver.Chrome(service=self.service, options=self.chrome_options)
        # base_url
        self.browser.set_page_load_timeout(30)
        self.base_url = 'https://twitter.com/search?q='
        self.number_tweets = 1500
        ### words to search in tweets..
        # 'raison', 'nouvelle', 'bravo', 'felicitations','aide','assistance','plaisir','confort',
        #                       'heureux', 'super','enthousiame','epnouissement','amour','cherir',
        #                       'adorer','faveur','foi', 'pute' , 'salope', 'pd', 'connasse', 'encule',

        self.words = ['attaque','traque','hante','ronge','bite', 'cul', 'connard', 'batard', 'enfoire', 'abruti gueule',\
                      'harcele', 'vilipende','chatie','blesse','bitch','nègre','salaud','sexe','putain','levrette',\
                      'homophobie','islam','nègro','imbécile','exclave','lgbt','gana']

        #self.browser.get('https://www.twitter.com')

    def get_tweets(self, keywords=[]):
        """
        :param self:
        :param queries: tweet which contains the word
        :return: DataFrame
        """
        # for k in self.words.keys():
        #     stop_loop_for_word = False
        #     for query in self.words[k]:
        #q = list()
        if not keywords:
            keywords = self.words
        for i, query in enumerate(keywords):
            tweets = []
            if query is not None and isinstance(query, str):
                #q.append(' OR '.join(query))
                query_encode = urllib.parse.quote(query) + urllib.parse.quote(' lang:fr')
                url = f"{self.base_url}{query_encode}&src=typed_query&f=live"

            try:
                # scroll management
                scrolling = True
                self.browser.get(url)
                while scrolling:

                    if WebDriverWait(self.browser, 40).until(
                            EC.presence_of_element_located((By.XPATH, '//div[@data-testid="cellInnerDiv"]'))):

                        e = self.browser.find_element(By.XPATH, '//article[@data-testid="tweet"]')

                        # read tweets
                        #print('----DEBUT----')
                        posts = self.get_tweet_data(e)
                        #print(posts)

                        if posts:
                            [tweets.append(tweet) for tweet in posts if tweet is not None and tweet not in tweets]
                        #print(tweets)
                        pd.DataFrame(data={'text': tweets}).to_csv(f'./data/{query}.csv', index=False)
                        #print("---FIN----")
                    # attempt
                    scroll_attempt = 0
                    try:
                        last_position = self.browser.execute_script("return document.body.scrollHeight;")
                    except exceptions.WebDriverException as e:
                        print(e.msg)
                        print('Une erreur vient de se produire : Note 1')
                        self.browser.close()
                        scrolling = False
                    while True:
                        # save data
                        if len(tweets) >= self.number_tweets:
                            scrolling = False
                            print('Taille atteinte pour :'+query)
                            #self.browser.close()
                            break

                        try:
                            # check scroll position
                            self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                            time.sleep(3)
                        except exceptions.WebDriverException as e:
                            print(e.msg)
                            scrolling = False
                            break
                        # new : return document.body.scrollHeight
                        # old : return window.pageYOffset
                        try:
                            curr_position = self.browser.execute_script("return document.body.scrollHeight ;")
                        except exceptions.WebDriverException as E:
                            print(E.msg)
                            print('Ne peut plus descendre.....')
                            scrolling = False
                            break
                        if last_position == curr_position:
                            scroll_attempt += 1
                            # end of scroll region
                            if scroll_attempt >= 2:
                                scrolling = False
                                print('Numbre de tentative atteint... ')
                                break
                            else:
                                time.sleep(2)
                        else:
                            last_position = curr_position
                            #self.browser.quit()
                            print('Scrolling ............')
                            #scrolling = False
                            break
                #self.browser.close()
            except StaleElementReferenceException as Exception:
                print(Exception.msg)
                self.browser.quit()

        #if i == len(keywords) - 1:
        self.browser.close()


    @staticmethod
    def get_tweet_data(card):
        """
        :param card:
        :return: list
        """
        try:
            if card is not None:
                el_texts = card.find_elements(by=By.XPATH, value='//div[@data-testid="tweetText"]')
                posts = [clean_text(tweet.text.strip()) for tweet in el_texts if el_texts is not None  and clean_text(tweet.text.strip()) ]
                #el_date = card.find_elements(by=By.XPATH, value='.//time')
                #dates = [dt.get_attribute('datetime') for dt in el_date if el_date is not None]
        except:
            return

        return posts


if __name__=='__main__':
    # utilisation de la classe pour scrapper les données avec selenium
    client = SeleniumClient()
    # utilisation de la méthode définie dans la classe SeleniumClient
    client.get_tweets()

    # regrouper les fichiers générés ev un seul fichier csv
    merge_file()
    # nettoyer les données du dataset
    clean_file()
    # build dataset # labeliser
    print('Build dataset and annotate')
    build_dataset()
    print('Create Model')
    # creer le model
    build_model()
