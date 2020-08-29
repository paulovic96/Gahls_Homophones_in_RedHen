import os
import pandas as pd
import numpy as np

CELEX_DIR = "/mnt/shared/corpora/Celex/" #"Data/"

def get_celex_filepath(lang="e", section="p", word="w"):
    assert (lang=="e" or lang=="g" or lang=="d"), "Celex supports English (lang='e'), german (lang='g'), and dutch (lang='d')."
    assert (section=="o" or section=="p" or section=="m" or section=="s" or section=="f"), "Celex contains data about orthogoraphy (section='o'), phonology (section='p'), morphology (section='m'), syntax (section='s'), and frequency  (section='f') of words."
    assert (word=="l" or word=="w"), "Celex contains data about lemmas (word='l') and wordforms (word='w')."
    lang_dir = "english" if lang=="e" else "german" if lang=="g" else "dutch"
    section_dir = lang+section+word
    file_path = os.path.join(CELEX_DIR, lang_dir, section_dir, section_dir+".cd")
    return file_path

def read_celex_file(lang="e", section="p", word="w"):
    """
    Returns dataframe with CELEX columns.

    See CELEX Guide for detailed description of column names.
    """
    df = pd.read_csv(get_celex_filepath(lang, section, word), header=None, sep="\n")
    df = df[0].str.split('\\', expand=True)
    if lang=="e" and section=="p" and word=="w":
        df = df.rename(columns={0: "IdNum", 1: "Word", 2: "Cob", 3: "IdNumLemma", 4: "PronCnt",
                                5: "PronStatus", 6: "PhonStrsDISC", 7: "PhonCVBr", 8: "PhonSylBCLX"})
        return df
    if lang=="e" and section=="s" and word=="l":
        df = df.rename(columns={0: "IdNum", 1: "Head", 2: "Cob", 3: "ClassNum", 4: "C_N", 
            5: "Unc_N", 6: "Sing_N", 7: "Plu_N", 8: "GrC_N", 9: "GrUnc_N", 10: "Attr_N", 
            11: "PostPos_N", 12: "Voc_N", 13: "Proper_N", 14: "Exp_N", 15: "Trans_V", 
            16: "TransComp_V", 17: "Intrans_V", 18: "Ditrans_V", 19: "Link_V", 20: "Phr_V", 
            21: "Prep_V", 22: "PhrPrep_V", 23: "Exp_V", 24: "Ord_A", 25: "Attr_A", 26: "Pred_A", 
            27: "PostPos_A", 28: "Exp_A", 29: "Ord_ADV", 30: "Pred_ADV", 31: "PostPos_ADV", 
            32: "Comb_ADV", 33: "Exp_ADV", 34: "Card_NUM", 35: "Ord_NUM", 36: "Exp_NUM", 
            37: "Pers_PRON", 38: "Dem_PRON", 39: "Poss_PRON", 40: "Refl_PRON", 41: "Wh_PRON", 
            42: "Det_PRON", 43: "Pron_PRON", 44: "Exp_PRON", 45: "Cor_C", 46: "Sub_C"})
    return df

def get_syl_counts(df):
    """
    Counts syllables using `PhonStrsDISC` column.

    `PhonStrsDISC` is a CELEX column that gives the words' pronunciations in DISC character set,
        with hyphens marking syllable boundaries, inverted commas showing points of primary stress,
        and double quotes showing points of secondary stress.

    Parameter: a dataframe that contains `PhonStrsDISC` column.
    Returns: a dataframe with syllable counts (`SylCnt` column).
    """
    df_copy = df.copy()
    df_copy["SylCnt"] = df_copy["PhonStrsDISC"].apply(lambda pron: pron.count("-")+1)
    return df_copy

def get_word_class(df):
    """
    Translates the `ClassNum` column to the corresponding word class.

    `ClassNum` is a CELEX column that gives the words' class (i.e., POS tag)

    Parameter: a dataframe that contains `ClassNum` column.
    Returns: the input dataframe with word classes (`Class` column).
    """
    df_copy = df.copy()
    num_to_class = {'1': 'N', '2': 'A', '3': 'NUM', '4': 'V', '5': 'ART', '6': 'PRON', 
                    '7': 'ADV', '8': 'PREP', '9': 'C', '10': 'I', '11': 'SCON', 
                    '12': 'CCON', '13': 'LET', '14': 'ABB', '15': 'TO'}
    df_copy["Class"] = df_copy["ClassNum"].apply(lambda num: num_to_class[num])
    return df_copy

def get_Noun_quotient(target_celex_merged_df, target_colname):
    """
    Computes Noun Quotient for target words
    from data in CELEX 
    """
    target_words = target_celex_merged_df[target_colname].unique()
    total_freq_df = target_celex_merged_df[[target_colname,'Cob_Wordform']].groupby([target_colname]).sum()
    byclass_freq_df = target_celex_merged_df[[target_colname,'Class','Cob_Wordform']].groupby([target_colname, 'Class']).sum()
    requested_multi_index = [(word, 'N') for word in target_words]
    count_df = byclass_freq_df.loc[requested_multi_index]['Cob_Wordform']/total_freq_df.loc[target_words]['Cob_Wordform']
    count_df = count_df.droplevel('Class')
    return_df = pd.DataFrame({'word': count_df.index, 'NQuot': count_df.values})
    return_df.loc[return_df.NQuot.isna(), "NQuot"] = 0.
    return return_df

#def get_Noun_quotient(target_celex_merged_df, target_colname):
#    """
#    Computes Noun Quotient for target words
#    from data in CELEX 
#    """
#    target_words = target_celex_merged_df[target_colname].unique()
#    total_freq_df = target_celex_merged_df[[target_colname,'Cob_Wordform']].groupby([target_colname]).sum()
#    byclass_freq_df = target_celex_merged_df[[target_colname,'Class','Cob_Wordform']].groupby([target_colname, 'Class']).sum()
#    index_level_set = set(byclass_freq_df.index.values)
#    requested_multi_index = [(word, 'N') for word in target_words if (word, 'N') in index_level_set]
#    valid_target_words = np.asarray(requested_multi_index)[:,0]
    
#    count_df = byclass_freq_df.loc[requested_multi_index]['Cob_Wordform'].astype(float)/total_freq_df.loc[valid_target_words]['Cob_Wordform'].astype(float)
#    count_df = count_df.droplevel('Class')
#    return_df = pd.DataFrame({'word': count_df.index, 'NQuot': count_df.values})
#    return_df.loc[return_df.NQuot.isna(), "NQuot"] = 0.
#    return return_df

def main(target_path, target_colname):
    # target words
    target_words_df = pd.read_csv(target_path)[target_colname]
    # CELEX data
    epw_df = get_syl_counts(read_celex_file(lang="e", section="p", word="w"))
    esl_df = get_word_class(read_celex_file(lang="e", section="s", word="l"))
    celex_merged_df = pd.merge(epw_df, esl_df, left_on='IdNumLemma', right_on='IdNum', suffixes=('_Wordform','_Lemma'))    
    celex_merged_df.Word = celex_merged_df.Word.str.lower()
    # get CELEX data for target words
    target_celex_merged_df = pd.merge(target_words_df, celex_merged_df, left_on = target_colname, right_on='Word')
    # compute whatever you want...
    return get_Noun_quotient(target_celex_merged_df, target_colname)

#df = main("/mnt/Restricted/Corpora/RedHen/homophone_analysis_scripts/hom.csv", "spell")


"""
def main():
    epw_df = get_syl_counts(read_celex_file(lang="e", section="p", word="w"))
    esl_df = get_word_class(read_celex_file(lang="e", section="s", word="l"))
    merged_df = pd.merge(epw_df, esl_df, left_on='IdNumLemma', right_on='IdNum', suffixes=('_Wordform','_Lemma'))
    return merged_df
"""
    # merged_df[merged_df.Word == "excuse"]
    #       IdNum_Wordform    Word Cob_Wordform IdNumLemma PronCnt PronStatus PhonStrsDISC      PhonCVBr   PhonSylBCLX  9  ... Dem_PRON Poss_PRON Refl_PRON Wh_PRON Det_PRON Pron_PRON Exp_PRON Cor_C Sub_C Class
    # 49468          30371  excuse          381      15518       2          P    Ik-'skjus  [VC][CCCVVC]  [Ik][skju:s]  S  ...        N         N         N       N        N         N        N     N     N     N
    # 49470          30372  excuse           59      15519       2          P    Ik-'skjuz  [VC][CCCVVC]  [Ik][skju:z]  S  ...        N         N         N       N        N         N        N     N     N     V
    # 49474         106134  excuse           59      15519       2          P    Ik-'skjuz  [VC][CCCVVC]  [Ik][skju:z]  S  ...        N         N         N       N        N         N        N     N     N     V
    # 49476         123334  excuse           59      15519       2          P    Ik-'skjuz  [VC][CCCVVC]  [Ik][skju:z]  S  ...        N         N         N       N        N         N        N     N     N     V
    # 49478         140244  excuse           59      15519       2          P    Ik-'skjuz  [VC][CCCVVC]  [Ik][skju:z]  S  ...        N         N         N       N        N         N        N     N     N     V

    # merged_df[merged_df.Word == "practice"]
    #        IdNum_Wordform      Word Cob_Wordform IdNumLemma PronCnt PronStatus PhonStrsDISC     PhonCVBr  PhonSylBCLX     9  ... Dem_PRON Poss_PRON Refl_PRON Wh_PRON Det_PRON Pron_PRON Exp_PRON Cor_C Sub_C Class
    # 105820          66704  practice         1702      35050       1          P    'pr{k-tIs  [CCVC][CVC]  [pr&k][tIs]  None  ...        N         N         N       N        N         N        N     N     N     N

