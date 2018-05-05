import pandas as pd
import numpy
from pylab import *
import urllib2
from bs4 import BeautifulSoup
import os

sourcePath = "E:\sentiment analisis\\"
data = []

# need to think about how to find the actual artical div
def read_files():
    for filename in os.listdir(sourcePath):
       data = pd.read_excel(sourcePath+filename)
       #links = data.loc['URLID','URLString','Title']
       for index, link in data.iterrows():
           if(index > 0):
                print(link['URLString'])
                response = urllib2.urlopen(link['URLString'])
                soup = BeautifulSoup(response, 'html.parser')
                title = soup.body.findAll(text=link['Title']).pop()
                print(title)
                article = title.find_next_sibling("div")
                print(article)

read_files()
