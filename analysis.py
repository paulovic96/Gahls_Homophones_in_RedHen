import preprocessing

filename = "Data/2016_all_words_no_audio.pickle"
hom_filename = "Data/hom.csv"

df = preprocessing.read_dataframe(filename, remove_pauses=False, remove_errors=True, preprocessing=True, drop_error_columns=False)

homophones_in_data, gahls_homophones, gahls_homophones_missing_in_data = preprocessing.read_and_extract_homophones(hom_filename, df)


source_file = '2016-12-17_1330_US_KCET_Asia_Insight'

test_hom_data = homophone_pairs_in_data[homophone_pairs_in_data.source_file == source_file].copy()
test_data = df[df.source_file == source_file].copy()

test_eaf_data = merged_annotation_gesture_eaf_data.copy()
test_eaf_data["duration"] = (test_eaf_data.end - test_eaf_data.start)/1000
test_eaf_data["annotation"] = test_eaf_data.annotation.apply(preprocessing.word_preprocessing)

words_in_data_not_annotated = list(set(list(np.unique(test_data.word))) - (set(list(np.unique(test_eaf_data.annotation)))))
words_annotated_not_in_data = list(set(list(np.unique(test_eaf_data.annotation))) - (set(list(np.unique(test_data.word)))))




# add gesture data
speech_annotation_eaf_data, gesture_eaf_data = read_eaf(filepath)
remove_pauses = True
merged_annotation_gesture_eaf_data = map_gestures_to_annotation(speech_annotation_eaf_data, gesture_eaf_data, remove_pauses=remove_pauses)

merged_annotation_gesture_eaf_data = binary_encode_gestures(merged_annotation_gesture_eaf_data, gesture_column = "gesture")






# pronunciation predictability
celex_dict_file = "Data/epw.cd"
import pandas as pd
import numpy as np
def get_english_phonology_from_celex(filename):
    phonology_dict = {"word":[], "disc":[], "clx":[]}
    with open(filename) as f:
        for line in f:
            line = line.strip().split("\\")
            word = line[1] # the word
            phonology_dict["word"].append(word)
            disc = line[6] # pronunciation in DISC notation, hyphens to mark syllable boundaries, inverted comma for primary stress and double quote for secondary stress (PhonStrsDISC)
            phonology_dict["disc"].append(disc)
            clx = line[8] # pronunciation in CELEX notation, with brackets (PhonSylBCLX)
            phonology_dict["clx"].append(clx)

    celex_phonology_dict = pd.DataFrame.from_dict(phonology_dict).drop_duplicates()
    celex_phonology_dict["disc_no_bound"] = celex_phonology_dict["disc"].apply(
        lambda x: x.replace("'", "").replace("-", ""))
    celex_phonology_dict["clx_no_bound"] = celex_phonology_dict["clx"].apply(
        lambda x: x.replace("[", "").replace("]", ""))
    return celex_phonology_dict

celex_phonology_dict = get_english_phonology_from_celex(celex_dict_file)

from g2p_en import G2p # https://github.com/Kyubyong/g2p

def get_ARPAbet_phonetic_transcription(word_list):
    g2p = G2p()
    arpabet_word_list = []
    for word in word_list:
        transcription = g2p(word)
        arpabet_word_list.append(transcription)

    return arpabet_word_list


def get_celex_transcription(df, celex_phonology_dict):

    return df.merge(celex_phonology_dict[["word", "disc", "clx", "disc_no_bound", "clx_no_bound"]], how = "left", left_on=["word", "celexPhon"], right_on=["word","disc_no_bound"])


berndt_character_code = pd.read_csv("Data/celex_phonetic_character_code_berndt1987.csv", delimiter=";")
berndt_conditional_probs = pd.read_csv("Data/Conditional_Probabilities_for_Grapheme-to-Phoneme_Correspondences_Berndt1987.csv",delimiter=";")

homophones_in_data = get_celex_transcription(homophones_in_data,celex_phonology_dict)


disc_characters_used_in_data = set(''.join(list(set(homophones_in_data.celexPhon))))
disc_characters_for_berndts_encoding = set(''.join([str(i) for i in list(set(berndt_character_code.DISC))]))


