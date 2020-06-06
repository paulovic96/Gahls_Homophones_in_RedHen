import os
import pandas as pd

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

