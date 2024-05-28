import yake
from yake import Yake
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from string import punctuation
import data
from kpminer import KPMiner

# Initialize Yake object
yake = Yake()

# Create an empty DataFrame to store the evaluation results
columns = ['ID', 'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F-Score']
eval_results_df = pd.DataFrame(columns=columns)

# Iterate through document IDs (assuming IDs range from 0 to 99)
for doc_id in range(5):
    # if doc_id == 59:
        # Initialize Yake object
        yake = Yake()
        kpminer = KPMiner()
        print('index', doc_id)
        judul, abstrak, golden = data.getData(doc_id)
        kata_kunci = golden.split(";")
        top_n = 50
        merge_ke = []

        # Combine title and abstract into one text
        teks_dataset = judul + ' ' + abstrak
        
        # Normalisasi dataset
        spesial = list(punctuation)
        spesial.append("`")
        spesial.append("``")
        spesial.append("''")
        dataset = []

        ke_yake = yake.keyword(teks_dataset, top_n)
        kandidat_yake = list(yake.getAllKeyword().keys())
        # print(len(kandidat_yake), kandidat_yake)

        ke_kpminer = kpminer.keyword(teks_dataset, top_n)
        kandidat_kpminer = list(kpminer.getAllKeyword().keys())
        # print(len(kandidat_kpminer), kandidat_kpminer)

        # Yake-KPMiner
        merge_ke = ke_yake.copy()
        temp_kpminer = ke_kpminer.copy()

        merge_kandidat = kandidat_yake.copy()
        
        new_value = top_n
        for key, value in merge_ke.items():
            merge_ke[key] = new_value
            new_value = new_value - 1

        new_value = top_n
        for key, value in temp_kpminer.items():
            temp_kpminer[key] = new_value
            new_value = new_value - 1

        # print("temp yake", merge_ke)
        # print("temp KP Miner", temp_kpminer)

        for kandidat in kandidat_kpminer:
            if kandidat not in merge_kandidat:
                merge_kandidat.append(kandidat)

        for key, value in temp_kpminer.items():
            if key in merge_ke:
                merge_ke[key] += value
            else:
                merge_ke[key] = value

        merge_ke_sort = dict(sorted(merge_ke.items(), key=lambda x: x[1], reverse=True))

        kandidat_yake_kpminer = merge_kandidat
        # print("kandidat_yake_kpminer", merge_ke_sort)
        ke_yake_kpminer = list(merge_ke_sort.keys())[:top_n]
        # print("ke_yake_kpminer", ke_yake_kpminer)

        keyphrases = ke_yake_kpminer
        candidates = kandidat_yake_kpminer
        # print("Jumlah Kandidat :", len(candidates))
        # print("Kandidat", candidates)
        print("Selected : ", keyphrases)
        

        # Perform evaluation
        kata_kunci_lower = [kata.lower().strip() for kata in kata_kunci]
        panjang_kata_kunci = len(kata_kunci_lower)
        all_kata = kata_kunci_lower + keyphrases

        vectorizer = CountVectorizer().fit_transform(all_kata)
        vectors = vectorizer.toarray()
        csim = cosine_similarity(vectors)

        masuk_threshold_temp = []
        for i in range(panjang_kata_kunci):
            idx = np.where(csim[i] > 0.8)
            for j in idx[0]:
                if j >= panjang_kata_kunci:
                    masuk_threshold_temp.append(all_kata[j])

        masuk_threshold = list(set(masuk_threshold_temp))
        print("masuk_threshold : ", masuk_threshold)
        print()

        masuk_tp = []
        for kata in kata_kunci:
            kata_final = kata.lower().strip()
            if kata_final in masuk_threshold:
                masuk_tp.append(kata_final)

        # print('masuk threshold ', masuk_threshold)
        # print('masuk tp : ', masuk_tp)

        TP = sum(kata.lower().strip() in masuk_threshold for kata in kata_kunci)
        FN = len(kata_kunci) - TP
        FP = len(masuk_threshold) - TP
        TN = len(candidates) - len(masuk_threshold)


        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0

        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0

        try:
            f_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f_score = 0

        # Store results in the DataFrame
        new_row = pd.DataFrame({
            'ID': [doc_id],
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'Precision': [precision],
            'Recall': [recall],
            'F-Score': [f_score]
        })

        # print(new_row)
        eval_results_df = pd.concat([eval_results_df, new_row], ignore_index=True)

# Display the evaluation results DataFrame
# print(eval_results_df)

# eval_results_df.to_csv(f"Eval Yake-KPMiner/yake-kpminer_{top_n}.csv")
