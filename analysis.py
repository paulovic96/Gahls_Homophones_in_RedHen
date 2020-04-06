import preprocessing

filename = "Data/2016_all_words_no_audio.pickle"
hom_filename = "Data/hom.csv"

df = preprocessing.read_dataframe(filename, remove_pauses=True, remove_errors=True, preprocessing=True, drop_error_columns=False)
homophone_pairs_in_data, homophones_in_data, gahls_homophones, gahls_homophones_in_data, gahls_homophones_missing_in_data = preprocessing.read_and_extract_homophones(hom_filename, df)



source_file = '2016-12-17_1330_US_KCET_Asia_Insight'

test_hom_data = homophone_pairs_in_data[homophone_pairs_in_data.source_file == source_file].copy()
test_data = df[df.source_file == source_file].copy()

test_eaf_data = merged_annotation_gesture_eaf_data.copy()
test_eaf_data["duration"] = (test_eaf_data.end - test_eaf_data.start)/1000
test_eaf_data["annotation"] = test_eaf_data.annotation.apply(preprocessing.word_preprocessing)

words_in_data_not_annotated = list(set(list(np.unique(test_data.word))) - (set(list(np.unique(test_eaf_data.annotation)))))
words_annotated_not_in_data = list(set(list(np.unique(test_eaf_data.annotation))) - (set(list(np.unique(test_data.word)))))


