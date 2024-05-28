import streamlit as st
import itertools
from yake import Yake
from kpminer import KPMiner

st.set_page_config(layout="wide")

st.header('Ekstraksi Kata Kunci Yake dan KP Miner')


class Gui:
    def __init__(self):
        self.judul = " "
        self.abstrak = " "

    def input(self):
        self.judul = st.text_input("Masukkan Judul :")
        self.abstrak = st.text_area("Masukkan Abstrak :")
        self.top_n = st.number_input("Jumlah Kata Kunci :", value=0, step=1)


# init kelas GUI
main = Gui()
input = main.input()

# init kelas Preprocessing
yake = Yake()
kpminer = KPMiner()

if st.button("Run"):
    if main.judul and main.abstrak:
        teks_dataset = main.judul + ' ' + main.abstrak

        ke_yake = yake.keyword(teks_dataset, main.top_n)
        ke_kpminer = kpminer.keyword(teks_dataset, main.top_n)

        merge_ke = ke_yake.copy()

        for key, value in ke_kpminer.items():
            if key in merge_ke:
                merge_ke[key] += value
            else:
                merge_ke[key] = value

        merge_ke_sort = dict(sorted(merge_ke.items(), key=lambda x: x[1], reverse=True))

        # ke_merge = list(merge_ke_sort)
        ke_merge = dict(itertools.islice(merge_ke_sort.items(), main.top_n))

        # st.write('token + stop word :',preprocessor.data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Kata Kunci Yake :**")
            for ke in (ke_yake.keys()):
                st.write(ke)
        with col2:
            st.write("**Kata Kunci KP Miner :**")
            for ke in (ke_kpminer.keys()):
                st.write(ke)
        with col3:
            st.write("**Kata Kunci Yake-KPMiner :**")
            for ke in (ke_merge.keys()):
                st.write(ke)

    else:
        st.warning("Masukkan Judul dan Abstrak terlebih dahulu.")
