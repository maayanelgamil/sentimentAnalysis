import pandas as pd
import numpy
from pylab import *
import urllib2
from bs4 import BeautifulSoup
import os

# Header to simulate that we request as a user
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}

sourcePath = "E:\sentiment analisis\\"

# need to think about how to find the actual artical div
def read_files():
    allData = pd.DataFrame()
    for filename in os.listdir(sourcePath):
       data = pd.read_excel(sourcePath+filename)
       allData = allData.append(data)
    return allData


def get_html_article_from_url(allData):
    # add the new file to the existing data we have
    for index, link in allData.iterrows():
            try:
                print(link['URLString'])
                req = urllib2.Request(link['URLString'], None, headers)
                response = urllib2.urlopen(req, timeout=5).read()
                soup = BeautifulSoup(response, 'html.parser')
                #title = soup.body.find_all(text=tmpTitle, recursive=True).pop().parent
                #print(title)
                #articleDivs = title.findNext("div")
                links = [e.get_text() for e in soup.body.find_all('p', recursive=True)]
                article = '\n'.join(links)
                print(article)
                allData['articl_dirty_text'] = article
            except Exception, e:
                print(e)
    writer = pd.ExcelWriter('allData.xlsx')
    allData.to_excel(writer, 'Sheet1')
    writer.save()

allData = read_files()
get_html_article_from_url(allData)