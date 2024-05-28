import pandas as pd
import nltk
import re

stopword = []
f = open("stopword.txt", "r")
for x in f:
    if x[-1:] == '\n':
        stopword.append(x[:-1])
    else:
        stopword.append(x)

dataset = "data/all_dataset.csv"
df = pd.read_csv(dataset)

def getData(dataset):
  judul = df.iloc[dataset, 0]
  abstrak = df.iloc[dataset, 1]
  keyphrases = df.iloc[dataset, 2]

  return judul, abstrak, keyphrases

def getAllData():
  all_data = []
  tokens_temp = []
  judul_temp = []
  abstrak_temp = []
  for id in range(100):
    judul, abstrak, keyphrases = getData(id)
    data = judul + ' ' + abstrak
    data = re.sub(r'[^\w\s-]', '', data)
    all_data.append(data.lower())

  return all_data