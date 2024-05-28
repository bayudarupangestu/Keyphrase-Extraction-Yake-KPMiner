import math
import string
import logging
from collections import Counter
import data
import re
import nltk
import itertools
from string import punctuation

import nltk
nltk.download('punkt')

stopword = []
f = open("stopword.txt", "r")
for x in f:
    if x[-1:] == '\n':
        stopword.append(x[:-1])
    else:
        stopword.append(x)


class KPMiner:
    def __init__(self):
        self.__lasf = 2
        self.__cutoff = 1000
        self.__alpha = 1.0
        self.__sigma = 3.0
        self.__candidates = []
        self.__special = list(punctuation)
        self.__clean_text = ""
        self.__weights = []
        self.__words_tokens = []

        self.__special.append("`")
        self.__special.append("``")
        self.__special.append("''")

    def keyword(self, text, n=100):
        # Cleansing
        self.__preProcesing(text)

        # Ekstraksi kandidat
        self.__candidateKeyphraseSelection()

        # Hitung bobot
        self.__candidateKeyphrasesWeightCalculation()

        # Hasil kata kunci
        kata_kunci = self.getAllKeyword()

        kata_kunci_sort = dict((sorted(kata_kunci.items(), key=lambda x: x[1], reverse=True)))

        if n > len(kata_kunci_sort):
            return kata_kunci_sort
        else:
            return dict(itertools.islice(kata_kunci_sort.items(), n))
    
    def __preProcesing(self, text):
        text = re.sub(r'[^\w\s-]', '', text)
        words = text.strip(string.punctuation)
        self.__clean_text = words.lower()
        self.__words_tokens = nltk.word_tokenize(self.__clean_text)

    def __candidateKeyphraseSelection(self):

        # Inisialisasi daftar kandidat
        candidates = []
        temp = []

        # Loop melalui setiap kata dalam teks
        for i, word in enumerate(self.__words_tokens):
            # Buat kandidat awal
            candidate = word.lower().strip(string.punctuation)

            # Filter kata yang terlalu pendek atau merupakan stop word
            if len(candidate) > 1 and candidate not in string.punctuation and candidate not in stopword:
                candidates.append(candidate)

        temp = []
        candidate_phrase = []

        for i, word in enumerate(self.__words_tokens):
            candidate = word.lower().strip(string.punctuation)

            if len(candidate) > 1:
                if candidate not in self.__special and candidate not in stopword:
                    temp.append(candidate)
                elif temp:
                    candidate_phrase.append(' '.join(temp))
                    temp = []
        
        if temp:
            candidate_phrase.append(' '.join(temp))
            temp = []

        new_candidate_phrase_2n = self.__sliding_window_ng(2, candidate_phrase)
        new_candidate_phrase_3n = self.__sliding_window_ng(3, candidate_phrase)

        for candidate in new_candidate_phrase_2n:
            candidates.append(candidate)

        for candidate in new_candidate_phrase_3n:
            candidates.append(candidate)
                
        # Hitung frekuensi kemunculan kandidat
        candidate_freq = Counter(candidates)

        # Lakukan filtering menggunakan LASF (Least Allowable Seen Frequency)
        candidates_filtered = [candidate for candidate, freq in candidate_freq.items() if freq >= self.__lasf]

        # Filter kandidat yang melewati batas cutoff
        self.__candidates = candidates_filtered.copy()

    def __candidateKeyphrasesWeightCalculation(self):
        position_factor = 1
        P_d = 0

        # Set P_d
        P_d += sum(1 for candidate in self.__candidates if len(candidate.split()) > 1)

        # Set N_d
        N_d = len(self.__candidates)

        # Set Boosting Factor
        b_f = self.__boostingFactor(N_d, P_d)

        for candidate in self.__candidates:
            tf = self.__tf(candidate)
            self.__weights.append(tf * b_f * position_factor)
    
    def __sliding_window_ng(self, n, kandidat):
        new_kandidat = []

        for k in kandidat:
            words = k.split()

            for i in range(len(words) - (n-1)):
                three_gram = ' '.join(words[i:i+n])
                new_kandidat.append(three_gram)

        return new_kandidat
    
    def __tf(self, kandidat):
        return self.__clean_text.count(kandidat)
    
    def __boostingFactor(self, N_d, P_d):
        return max(N_d / (P_d * self.__alpha), self.__sigma)
    
    def getTokenisasi(self):
        return self.__words_tokens
    
    def getCandidateKeyphraseSelection(self):
        return self.__candidates
    
    def getCandidateKeyphraseWeight(self):
        candidates_weight = dict(zip(self.__candidates, self.__weights))
        return candidates_weight
    
    def getAllKeyword(self):
        all_keyword = dict(zip(self.__candidates, self.__weights))
        return all_keyword
