
import preprocessing

filename = "Data/2016_all_words_no_audio.pickle"
hom_filename = "Data/hom.csv"

df = preprocessing.read_dataframe(filename, remove_pauses=True, remove_errors=True, preprocessing=True, drop_error_columns=False)
homophone_pairs_in_data, homophones_in_data, gahls_homophones, gahls_homophones_in_data, gahls_homophones_missing_in_data = preprocessing.read_and_extract_homophones(hom_filename, df)

