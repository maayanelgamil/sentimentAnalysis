import string
from nltk.corpus import stopwords
import re, string
import pandas as pd
import matplotlib.pyplot as plt


# Define regex consts
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)',  # anything else
]

# we defined emoticoned regex to identify nd clean emoticons from our data
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


# Create stop word dictionary
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation


def clean_stopwords(text):
    no_stopwords_tokens = []

    # Remove stop words
    for token in text:
        if token not in stop:
            no_stopwords_tokens.append(token)

    return no_stopwords_tokens

def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*', r'', s)
    return tokens_re.findall(s)

def preprocess(s):
    tokens = tokenize(s)
    # To lower
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def clearup(s, chars):
    return re.sub('[%s]' % chars, '', s).lower()

def clean(data):
    # Drop empty classification or empty article
    # python doens't recognize empty string as NA
    # therefor we'll replace all empty string before using dropna
    data['classification'].replace('', pd.np.nan, inplace=True)
    data['articl_dirty_text'].replace('', pd.np.nan, inplace=True)
    data.dropna(subset=['classification'], inplace=True)
    data.dropna(subset=['articl_dirty_text'], inplace=True)

    # now we'll remove all the irrelevent classifications
    data.drop(data[data.classification == 'not relevant'].index, inplace=True)


    print('irrelevant rows removed')
    print("now lets clean the text")

    row_it = data.iterrows()
    text_clean = []

    # Iterate the data clean it, and return it back to the data data frame
    for i, line in row_it:
        no_stopwords_tokens = []
        tokens = preprocess(line['articl_dirty_text'])
        no_stopwords_tokens = clean_stopwords(tokens)
        # create line of the text
        cleanText = ' '.join(no_stopwords_tokens)
        # remove title from the actual article
        cleanText = cleanText.replace(line['Title'], "")
        # remove numbers from the textx
        cleanText = clearup(cleanText, string.digits)
        text_clean.append(cleanText)

    data['text_clean'] = text_clean

    #we can now explore the data classifications
    # Explore gender distributation count
    print(data.classification.value_counts())

    # Explore distributation of words per gender
    positive = data[data['classification'] == 'positive']
    negative = data[data['classification'] == 'negative']
    neutral = data[data['classification'] == 'neutral']
    positive_Words = pd.Series(' '.join(positive['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    negative_Words = pd.Series(' '.join(negative['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    neutral_Words = pd.Series(' '.join(neutral['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    All_words = pd.Series(' '.join(data['text_clean'].astype(str)).lower().split(" ")).value_counts()[:10]

    print("**********FINISHED CLEANING THE TEXT***************")
    print(positive_Words)
    ts = positive_Words.plot(kind='bar', stacked=True, colormap='hot')
    ts.plot()
    plt.show()
    print(negative_Words)
    ts = negative_Words.plot(kind='bar', stacked=True, colormap='plasma')
    ts.plot()
    plt.show()
    print(neutral_Words)
    ts = neutral_Words.plot(kind='bar', stacked=True, colormap='plasma')
    ts.plot()
    plt.show()
    print("**ALL WORDS**")
    print(All_words)
    ts = All_words.plot(kind='bar', stacked=True, colormap='Paired')
    ts.plot()
    plt.show()
    return data


