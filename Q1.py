import pandas as pd
from nltk.corpus import stopwords
import re, string
from nltk.stem import PorterStemmer

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
    ps = PorterStemmer()
    # Remove stop words and stem
    for token in text:
        if token not in stop:
            token = ps.stem(token)
            no_stopwords_tokens.append(token)

    return no_stopwords_tokens


def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*', r'', s)
    return tokens_re.findall(s)


def preprocess(s, title):
    # remove title from the actual article
    cleanText = s.replace(title, "")
    # remove numbers from the textx
    cleanText = clearup(cleanText, string.digits)
    tokens = tokenize(cleanText)
    # To lower
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def clearup(s, chars):
    return re.sub('[%s]' % chars, '', s).lower()


def clean(data):
    title_column_name = "title_clean"
    article_column_name = "text_clean"
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

    # Iterate the article_dirty_text clean it, and return it back to the data data frame
    for i, line in row_it:
        no_stopwords_tokens = []
        tokens = preprocess(line['articl_dirty_text'], line['Title'])
        no_stopwords_tokens = clean_stopwords(tokens)
        # create line of the text
        cleanText = ' '.join(no_stopwords_tokens)
        text_clean.append(cleanText)

    data[article_column_name] = text_clean

    row_it = data.iterrows()
    title_clean = []

    # Iterate the Title dirty text clean it, and return it back to the data data frame
    for i, line in row_it:
        no_stopwords_tokens = []
        tokens = preprocess(line['Title'], "")
        no_stopwords_tokens = clean_stopwords(tokens)
        # create line of the text
        cleanTitle = ' '.join(no_stopwords_tokens)
        title_clean.append(cleanTitle)

    data[title_column_name] = title_clean

    print(data.classification.value_counts())

    return data


