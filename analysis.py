import preprocessing
import pandas as pd
import numpy as np

filename = "Data/2016_all_words_no_audio.pickle"
hom_filename = "Data/hom.csv"

#df = preprocessing.read_dataframe(filename, remove_pauses=True, remove_errors=True, preprocessing=True, drop_error_columns=False)

sub_df = pd.read_csv("sub_df.csv", index_col="Unnamed: 0")
eaf_data = preprocessing.get_additional_data_from_files(sub_df, "eaf")
seg_data = preprocessing.get_additional_data_from_files(sub_df, "seg")
gentle_data = preprocessing.get_additional_data_from_files(sub_df, "gentle")
video_data = preprocessing.get_additional_data_from_files(sub_df, "video")


homophones_in_data, gahls_homophones, gahls_homophones_missing_in_data = preprocessing.read_and_extract_homophones(hom_filename, df)
homophones_in_data.sort_values(by = ["source_file", "start"], inplace = True)


eaf_hom_data = eaf_data[eaf_data['annotation'].isin(np.unique(homophones_in_data["word"]))].sort_values(by = ["source_file", "start"])
video_hom_data = video_data[video_data['word'].isin(np.unique(homophones_in_data["word"]))].sort_values(by = ["source_file", "start"])


