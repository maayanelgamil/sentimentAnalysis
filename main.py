import pandas as pd
import numpy
from pylab import *
import urllib2
from bs4 import BeautifulSoup
import os

sourcePath = "E:\sentiment analisis\\"
data = []

def read_files():
    for filename in os.listdir(sourcePath):
       print(filename)
       data = pd.read_excel(sourcePath+filename)

read_files()
