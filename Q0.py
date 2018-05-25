import pandas as pd
import numpy as np
from pylab import *
import urllib.request as urllib
from bs4 import BeautifulSoup
import os

# this file contains the methods that read the articles from the given urls
# and saves these articls to our local .xlsx file for developent convinient


# Header to simulate that we request as a user
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}

sourcePath = "E:\sentiment analisis\\"
excelPath = "./allData.xlsx"
article_column = []

# get the files from the assignment's excel
def read_files():
    allData = pd.DataFrame()
    for filename in os.listdir(sourcePath):
       data = pd.read_excel(sourcePath+filename)
       allData = allData.append(data)
    return allData


# iterate each url and extract its content
def get_html_article_from_url(allData):
    # add the new file to the existing data we have
    for index, link in allData.iterrows():
            try:
                print(link['URLString'])
                if(link['classification'] == "" or pd.isnull(link['classification'])):
                    article_column.append("")
                    continue;
                req = urllib.Request(link['URLString'], None, headers)
                response = urllib.urlopen(req, timeout=5).read()
                soup = BeautifulSoup(response, 'html.parser')
                links = [e.get_text() for e in soup.body.find_all('p', recursive=True)]
                article = '\n'.join(links)
                print(article)
                article_column.append(article)
            except Exception as e:
                article_column.append("")
                print(e)
    allData['articl_dirty_text'] = article_column
    writer = pd.ExcelWriter('allData.xlsx')
    allData.to_excel(writer, 'Sheet1')
    writer.save()

# read the raw data xlxs
def read_excel():
    return pd.read_excel(excelPath)


def get_raw_data():
    # allData = read_files()
    # get_html_article_from_url(allData)
    data = read_excel()
    return read_excel()