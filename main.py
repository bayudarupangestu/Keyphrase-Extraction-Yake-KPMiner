import yake
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

yake_obj = yake.Yake()


all_dataset = pd.read_csv('./data/all_dataset.csv')

# teks_dataset = all_dataset['Judul'][0] + all_dataset['Abstrak'][0] + " "

# hasil = yake_obj.keyword(teks_dataset, int(20))

# keyphrase = list(hasil.keys())

# print("keyphrase", keyphrase)
# print('len ', len(keyphrase))

all_data = []

for index, row in all_dataset.iterrows():
  abstrak = row['Abstrak'].strip() + " "
  judul = row['Judul'].strip()

  teks_dataset = judul +". " + abstrak.strip()
  # print(teks_dataset)

  
  print('index-', index)
  # print(teks_dataset)
  hasil = yake_obj.keyword(teks_dataset, int(500))
  all_data.append(hasil)

print(all_data[0])

