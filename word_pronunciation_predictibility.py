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


