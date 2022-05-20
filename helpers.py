import os
import re
import glob
import pandas as pd
from detoxify import Detoxify


"""
template twitter
"""
dict = {
    '\\u00e0': 'a',
    '\\u00e2': 'a',
    '\\u00e4': 'a',
    '\\u00C3': 'A',
    '\\u00C2': 'A',
    '\\u00C1': 'A',
    '\\u00C0': 'A',
    '\\u00c0': 'A',
    '\\u00e7': 'c',
    '\\u00c7': 'C',
    '\\u00e8': 'e',
    '\\u00e9': 'e',
    '\\u00ea': 'e',
    '\\u00eb': 'e',
    '\\u00c8': 'E',
    '\\u00C9': 'E',
    '\\u00c9': 'E',
    '\\u00CA': 'E',
    '\\u00CB': 'E',
    '\\u00ee': 'i',
    '\\u00ef': 'i',
    '\\u00ce': 'I',
    '\\u00CC': 'I',
    '\\u00CD': 'I',
    '\\u00CE': 'I',
    '\\u00CF': 'I',
    '\\u00f4': 'o',
    '\\u00f6': 'o',
    '\\u00D2': 'O',
    '\\u00D3': 'O',
    '\\u00D4': 'O',
    '\\u00D5': 'O',
    '\\u00D6': 'O',
    '\\u00f9': 'u',
    '\\u00fb': 'u',
    '\\u00fc': 'u',
    '\\u00D9': 'U',
    '\\u00DA': 'U',
    '\\u00DB': 'U',
    '\\u00DC': 'U',
    '\\u2019': "'",
    '\\u2018': "'",
    '\\u20ac': "euros",
    '&lt;': 'inferieur',
    '&le;': 'inferieur egale',
    '&gt;': 'superieur',
    '&ge;': 'superieur egale',
    '\\u00a0': '',
    '\\u00ab': "'",
    '\\u00bb': "'",
    ',': ' ',
    '\\ud83d': '',
    '\\ude09': '',
    '\\ude18': '',
    '\\ude08': '',
    '\n': ' ',
    '\r': '',
    '//': '',
    '"': '',
    '*': '',
    '(': '',
    ')': '',
    '?': '',
    '.': '',
    '!': '',
    '*' : '',
    '+' : '',
    '-' : '',
    '[' : '',
    ']' : '',
    '{' : '',
    '}' : ''

}

# define the model
model = Detoxify('multilingual')


def mkdirs(dirs):
    """
    @:param dirs Create the dirs if not exists
    @:return None
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def clean_text(text):
    """
    :param text:
    :return: str
    """
    for i, j in dict.items():
        text = text.replace(i,j)
    # remove # (hashtag)
    text = re.sub('#','', text)
    # remove user profile twitter: @xxxxx
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    # remove http. links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)
    # remove numbers
    text = re.sub(r"\d\S+",'',text)
    # lower
    text = text.lower()
    # remove spaces
    text = text.strip()
    text = re.sub(' +', ' ', text)

    return text


def merge_file():
    """
    merge all csv files into one
    """
    csv_files = glob.glob('./data/*.csv')
    df = pd.DataFrame(columns=['text'])
    for csv_file in csv_files:
        temp = pd.read_csv(csv_file)
        os.remove(csv_file)
        df = pd.concat([df, temp], ignore_index=True)
    return df.to_csv('./data/merge_csv.csv', index=False)


def clean_file():
    """
    clean merged dataset using clean_text function
    """
    file = './data/merge_csv.csv'
    df = pd.read_csv(file)
    df['text'] = df['text'].apply(clean_text)
    os.remove(file)
    df.to_csv(file, index=False)


def label(tweet):
    """
    labelize the tweet
    :return: prediction
    """
    res = model.predict(tweet)
    max_key = max(res, key=res.get)
    if res[max_key] <= 0.3:
        return 'no_toxic'
    return 'toxic'


def build_dataset():
    # build dataset
    df = pd.read_csv('./data/merge_csv.csv')
    df['label'] = df['text'].apply(label)
    df.to_csv('./data/dataset.csv', index=False)



