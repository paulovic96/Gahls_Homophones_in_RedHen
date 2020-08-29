import preprocessing
import pandas as pd
import numpy as np
import importlib
from english_contractions import ENGLISH_CONTRACTIONS
import merging_dataframes
import word_pronunciation_predictibility
import celex_files

celex_dict_file = "/mnt/shared/corpora/Celex/english/epw/epw.cd"
filename = "/mnt/Restricted/Corpora/RedHen/2016_all_words_no_audio.pickle"
hom_filename = "/mnt/Restricted/Corpora/RedHen/hom.csv"
berndt_character_coding_file = "/mnt/Restricted/Corpora/RedHen/phonetic_character_code_berndt1987.csv"
berndt_conditional_probs_file = "/mnt/Restricted/Corpora/RedHen/Conditional_Probabilities_for_Grapheme-to-Phoneme_Correspondences_Berndt1987.csv"

homophones_in_data_celex_merged = pd.read_csv("2016_all_words_no_audio_homophones.csv", index_col = "Unnamed: 0")

video_data = preprocessing.get_additional_data_from_files(homophones_in_data_celex_merged, "video") # only for homophones

video_data.to_csv("2016_all_words_no_audio_video_data.csv")