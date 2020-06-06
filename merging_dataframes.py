import pandas as pd
import numpy as np
from english_contractions import ENGLISH_CONTRACTIONS


def merge_eaf_df_to_homophone_data(hom_in_data_df, eaf_df):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param eaf_df (pd.DataFrame): The dataframe containing additional data from the eaf files
    :return (pd.DataFrame): merged dataframe containing additional data:
    - gesture : string with gestures present in video during articulation
    - HandMoving', 'PersonOnScreen','SpeakerOnScreen',
      'HeadMoving/MovingVertically','ShoulderMoving/NotWithHead',
      'HeadMoving/MovingHorizontally','ShoulderMoving/NoSlidingWindow',
      'none', 'ShoulderMoving/SlidingWindow',: one hat encoded gestures
    - is_gesture : indicating whether a gesture is present in video
    """
    hom_in_eaf_data_df = eaf_df[eaf_df['annotation'].isin(np.unique(hom_in_data_df["word"]))].sort_values(by = ["source_file", "start"]) # extract homophones and sort by file and start
    #hom_in_eaf_data_df.sort_values(by = ["source_file", "start"], inplace = True)
    hom_in_data_df.start = hom_in_data_df.start.round(2) # round time to match when merged
    hom_in_eaf_data_df.start = (hom_in_eaf_data_df.start / 1000).round(2) # convert and round time
    hom_in_eaf_data_df.end = (hom_in_eaf_data_df.end / 1000).round(2)
    hom_in_eaf_data_df.rename(columns={"annotation": "word"}, inplace=True)

    return hom_in_data_df.merge(hom_in_eaf_data_df[['word', 'source_file', 'start', 'gesture', 'HandMoving', 'PersonOnScreen',
                      'SpeakerOnScreen', 'HeadMoving/MovingVertically',
                      'ShoulderMoving/NotWithHead', 'HeadMoving/MovingHorizontally',
                      'ShoulderMoving/NoSlidingWindow', 'none',
                      'ShoulderMoving/SlidingWindow', 'is_gesture']], on=["source_file", "start", "word"], how="left")


def merge_video_df_to_homophone_data(hom_in_data_df, video_df):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param video_df (pd.DataFrame): The dataframe containing additional data from the mp4 files
    :return (pd.DataFrame): merged dataframe containing additional data:
    - video_snippet_size : size of the video snippet during which the word is articulated
    """
    hom_in_video_df = video_df[video_df['word'].isin(np.unique(hom_in_data_df["word"]))].sort_values(
        by=["source_file", "start"])
    hom_in_video_df.sort_values(by=["source_file", "start"], inplace=True)
    hom_in_data_df.start = hom_in_data_df.start.round(2)
    hom_in_video_df.start = hom_in_video_df.start.round(2)

    return hom_in_data_df.merge(hom_in_video_df[['source_file', 'word', 'start', 'video_snippet_size']], on =["source_file","start", "word"], how="left")

def merge_gentle_df_to_homophone_data(hom_in_data_df, gentle_df):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param gentle_df (pd.DataFrame): The dataframe containing additional data from the gentle files
    :return (pd.DataFrame): merged dataframe containing additional data:
    - gentle_prev_word : the next word according to the gentle file
    - gentle_next_word : the prev word according to the gentle file
    - gentle_end_of_sentence : boolean whether the word is the end of the sentence
    - gentle_start_of_sentence : boolean whether the word is the start of the sentence
    - gentle_preceding_marker: boolean whether the word preceding a marker
    - gentle_subsequent_marker: boolean whether the word subsequent a marker
    - gentle_merging : information about how confident we are about the merging
    - gentle_index : idx position of the word in the gentle transcript as proxy for the start time
    """
    gentle_df["IDX"] = gentle_df.index  # no start time information included
    gentle_df["word"] = gentle_df.word.str.lower()
    gentle_df["prev_word"] = gentle_df.prev_word.str.lower()
    gentle_df["next_word"] = gentle_df.next_word.str.lower()

    hom_in_gentle_data_df = gentle_df[gentle_df['word'].isin(np.unique(hom_in_data_df["word"]))].sort_values(by = ["source_file", "IDX"]) # extract homophones and sort by file and index

    merged_hom_data = None # The dataframe we are building by incrementally merging the dataframes
    for file in np.unique(hom_in_data_df.source_file): # look at sub df for each file separately
        hom_in_data_i = hom_in_data_df[hom_in_data_df.source_file == file].sort_values(by=["start"]) # make sure that df is sorted
        hom_in_gentle_data_i = hom_in_gentle_data_df[hom_in_gentle_data_df.source_file == file].sort_values(by=["IDX"]) # make sure that the gentle data for this file is sorted

        # add columns
        hom_in_data_i["gentle_prev_word"] = None
        hom_in_data_i["gentle_next_word"] = None
        hom_in_data_i["gentle_end_of_sentence"] = None
        hom_in_data_i["gentle_start_of_sentence"] = None
        hom_in_data_i["gentle_preceding_marker"] = None
        hom_in_data_i["gentle_subsequent_marker"] = None
        hom_in_data_i["gentle_merging"] = None
        hom_in_data_i["gentle_index"] = None

        for row_index, row in hom_in_data_i.iterrows(): #iterate over each row
            hom = row["word"] #the homophones we are looking at

            prev_word = row["prev_word"] # context
            next_word = row["next_word"]

            if prev_word in list(ENGLISH_CONTRACTIONS.keys()): #make sure that no error occures because of contractions
                prev_word = ENGLISH_CONTRACTIONS[prev_word].split(" ")[-1]
            if next_word in list(ENGLISH_CONTRACTIONS.keys()):
                next_word = ENGLISH_CONTRACTIONS[next_word].split(" ")[0]

            if isinstance(prev_word, str): # otherwise None
                # some words in the gentle file are splitted by "-" or  "'"
                if "-" in prev_word:
                    prev_word = prev_word.split("-")[-1] # the last part of the previous
                if "'" in prev_word:
                    prev_word = prev_word.split("'")[0] # the word "stem"
            if isinstance(next_word, str):
                if "-" in next_word:
                    next_word = next_word.split("-")[0] # the first part for next
                if "'" in next_word:
                    next_word = next_word.split("'")[0]

            possible_matches = hom_in_gentle_data_i[hom_in_gentle_data_i.word == hom].index # all positions on which the homophone occurs

            for idx in possible_matches: #iterating over possible matches (still sorted by occurence)
                possible_row = hom_in_gentle_data_i.loc[idx] # get corresponding row

                possible_prev_word = possible_row.prev_word
                possible_next_word = possible_row.next_word

                if possible_prev_word in list(ENGLISH_CONTRACTIONS.keys()): #make sure that no error occures because of contractions
                    possible_prev_word = ENGLISH_CONTRACTIONS[possible_prev_word].split(" ")[-1]
                if possible_next_word in list(ENGLISH_CONTRACTIONS.keys()):
                    possible_next_word = ENGLISH_CONTRACTIONS[possible_next_word].split(" ")[0]

                if isinstance(possible_prev_word, str): # same preprocessing as above
                    if "-" in possible_prev_word:
                        possible_prev_word = possible_prev_word.split("-")[-1]
                    if "'" in possible_prev_word:
                        possible_prev_word = possible_prev_word.split("'")[0]
                if isinstance(possible_next_word, str):
                    if "-" in possible_next_word:
                        possible_next_word = possible_next_word.split("-")[0]
                    if "'" in possible_next_word:
                        possible_next_word = possible_next_word.split("'")[0]

                if possible_prev_word == prev_word and possible_next_word == next_word: # high-confidence match: word, prev and next are all equal
                    hom_in_data_i.at[row_index, "gentle_prev_word"] = possible_row.prev_word
                    hom_in_data_i.at[row_index, "gentle_next_word"] = possible_row.next_word
                    hom_in_data_i.at[row_index, 'gentle_merging'] = "high-confidence"
                    hom_in_data_i.at[row_index, "gentle_end_of_sentence"] = possible_row.end_of_sentence
                    hom_in_data_i.at[row_index, "gentle_start_of_sentence"] = possible_row.start_of_sentence
                    hom_in_data_i.at[row_index, "gentle_preceding_marker"] = possible_row.preceding_marker
                    hom_in_data_i.at[row_index, "gentle_subsequent_marker"] = possible_row.subsequent_marker
                    hom_in_data_i.at[row_index, "gentle_index"] = possible_row.IDX
                    hom_in_gentle_data_i.drop(index=idx, inplace=True)
                    break

                elif possible_prev_word == prev_word or possible_next_word == next_word: # low-confidence match: at least one of prev or next word is equal
                    # print(possible_row.prev_word, possible_row.word, possible_row.next_word)
                    hom_in_data_i.at[row_index, "gentle_prev_word"] = possible_row.prev_word
                    hom_in_data_i.at[row_index, "gentle_next_word"] = possible_row.next_word
                    hom_in_data_i.at[row_index, 'gentle_merging'] = "low-confidence"
                    hom_in_data_i.at[row_index, "gentle_end_of_sentence"] = possible_row.end_of_sentence
                    hom_in_data_i.at[row_index, "gentle_start_of_sentence"] = possible_row.start_of_sentence
                    hom_in_data_i.at[row_index, "gentle_preceding_marker"] = possible_row.preceding_marker
                    hom_in_data_i.at[row_index, "gentle_subsequent_marker"] = possible_row.subsequent_marker
                    hom_in_data_i.at[row_index, "gentle_index"] = possible_row.IDX
                    hom_in_gentle_data_i.drop(index=idx, inplace=True)
                    break

        if merged_hom_data is None:
            merged_hom_data = hom_in_data_i # first file data replaces None
        else:
            merged_hom_data = pd.concat([merged_hom_data, hom_in_data_i]) # Concatenate already merged files with new file data

    merged_hom_data.replace(to_replace=[None], value=np.nan, inplace=True)
    return merged_hom_data



def merge_seg_df_to_homophone_data(hom_in_data_df, seg_df):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param seg_df (pd.DataFrame): The dataframe containing additional data from the seg files
    :return (pd.DataFrame): merged dataframe containing additional data:
    - seg_prev_word : the next word according to the gentle file
    - seg_next_word : the prev word according to the gentle file
    - seg_end_of_sentence : boolean whether the word is the end of the sentence
    - seg_start_of_sentence : boolean whether the word is the start of the sentence
    - seg_preceding_marker: boolean whether the word preceding a marker
    - seg_subsequent_marker: boolean whether the word subsequent a marker
    - seg_merging : information about how confident we are about the merging
    - seg_index : idx position of the word in the seg transcript as proxy for the start time
    - pos : part of speech
    - rel1 : relation 1
    - rel2 : relation 2
    - lemma : lemma
    """
    seg_df["IDX"] = seg_df.index
    seg_df["word"] = seg_df.word.str.lower()
    seg_df["prev_word"] = seg_df.prev_word.str.lower()
    seg_df["next_word"] = seg_df.next_word.str.lower()

    hom_in_seg_data_df = seg_df[seg_df['word'].isin(np.unique(hom_in_data_df["word"]))].sort_values(by=["source_file", "IDX"])  # extract homophones and sort by file and index

    merged_hom_data = None
    for file in np.unique(hom_in_data_df.source_file):
        hom_in_data_i = hom_in_data_df[hom_in_data_df.source_file == file].sort_values(by=["start"])
        hom_in_seg_data_i = hom_in_seg_data_df[hom_in_seg_data_df.source_file == file].sort_values(by=["IDX"])

        hom_in_data_i["seg_prev_word"] = None
        hom_in_data_i["seg_next_word"] = None
        hom_in_data_i["seg_end_of_sentence"] = None
        hom_in_data_i["seg_start_of_sentence"] = None
        hom_in_data_i["seg_preceding_marker"] = None
        hom_in_data_i["seg_subsequent_marker"] = None
        hom_in_data_i["seg_merging"] = None
        hom_in_data_i["seg_index"] = None
        hom_in_data_i["pos"] = None
        hom_in_data_i["rel1"] = None
        hom_in_data_i["rel2"] = None
        hom_in_data_i["lemma"] = None

        for row_index, row in hom_in_data_i.iterrows():
            hom = row["word"]
            prev_word = row["prev_word"]
            next_word = row["next_word"]

            if prev_word in list(ENGLISH_CONTRACTIONS.keys()):
                prev_word = ENGLISH_CONTRACTIONS[prev_word].split(" ")[-1]
            if next_word in list(ENGLISH_CONTRACTIONS.keys()):
                next_word = ENGLISH_CONTRACTIONS[next_word].split(" ")[0]

            if isinstance(prev_word, str):
                if "-" in prev_word:
                    prev_word = prev_word.split("-")[-1]
                if "'" in prev_word:
                    prev_word = prev_word.split("'")[0]
            if isinstance(next_word, str):
                if "-" in next_word:
                    next_word = next_word.split("-")[0]

                if "'" in next_word:
                    next_word = next_word.split("'")[0]

            possible_matches = hom_in_seg_data_i[hom_in_seg_data_i.word == hom].index

            for idx in possible_matches:
                possible_row = hom_in_seg_data_i.loc[idx]
                possible_prev_word = possible_row.prev_word
                possible_next_word = possible_row.next_word

                if possible_prev_word in list(ENGLISH_CONTRACTIONS.keys()):
                    possible_prev_word = ENGLISH_CONTRACTIONS[possible_prev_word].split(" ")[-1]
                if possible_next_word in list(ENGLISH_CONTRACTIONS.keys()):
                    possible_next_word = ENGLISH_CONTRACTIONS[possible_next_word].split(" ")[0]

                if isinstance(possible_prev_word, str):
                    if "-" in possible_prev_word:
                        possible_prev_word = possible_prev_word.split("-")[-1]
                    if "'" in possible_prev_word:
                        possible_prev_word = possible_prev_word.split("'")[0]
                if isinstance(possible_next_word, str):
                    if "-" in possible_next_word:
                        possible_next_word = possible_next_word.split("-")[0]
                    if "'" in possible_next_word:
                        possible_next_word = possible_next_word.split("'")[0]

                if possible_prev_word == prev_word and possible_next_word == next_word:
                    hom_in_data_i.at[row_index, "seg_prev_word"] = possible_row.prev_word
                    hom_in_data_i.at[row_index, "seg_next_word"] = possible_row.next_word
                    hom_in_data_i.at[row_index, 'seg_merging'] = "high-confidence"
                    hom_in_data_i.at[row_index, "seg_end_of_sentence"] = possible_row.end_of_sentence
                    hom_in_data_i.at[row_index, "seg_start_of_sentence"] = possible_row.start_of_sentence
                    hom_in_data_i.at[row_index, "seg_preceding_marker"] = possible_row.preceding_marker
                    hom_in_data_i.at[row_index, "seg_subsequent_marker"] = possible_row.subsequent_marker
                    hom_in_data_i.at[row_index, "seg_index"] = possible_row.IDX
                    hom_in_data_i.at[row_index, "pos"] = possible_row.pos
                    hom_in_data_i.at[row_index, "rel1"] = possible_row.rel1
                    hom_in_data_i.at[row_index, "rel2"] = possible_row.rel2
                    hom_in_data_i.at[row_index, "lemma"] = possible_row.lemma

                    hom_in_seg_data_i.drop(index=idx, inplace=True)
                    break

                elif possible_prev_word == prev_word or possible_next_word == next_word:

                    hom_in_data_i.at[row_index, "seg_prev_word"] = possible_row.prev_word
                    hom_in_data_i.at[row_index, "seg_next_word"] = possible_row.next_word
                    hom_in_data_i.at[row_index, 'seg_merging'] = "low-confidence"
                    hom_in_data_i.at[row_index, "seg_end_of_sentence"] = possible_row.end_of_sentence
                    hom_in_data_i.at[row_index, "seg_start_of_sentence"] = possible_row.start_of_sentence
                    hom_in_data_i.at[row_index, "seg_preceding_marker"] = possible_row.preceding_marker
                    hom_in_data_i.at[row_index, "seg_subsequent_marker"] = possible_row.subsequent_marker
                    hom_in_data_i.at[row_index, "seg_index"] = possible_row.IDX
                    hom_in_data_i.at[row_index, "pos"] = possible_row.pos
                    hom_in_data_i.at[row_index, "rel1"] = possible_row.rel1
                    hom_in_data_i.at[row_index, "rel2"] = possible_row.rel2
                    hom_in_data_i.at[row_index, "lemma"] = possible_row.lemma
                    hom_in_seg_data_i.drop(index=idx, inplace=True)
                    break

        if merged_hom_data is None:
            merged_hom_data = hom_in_data_i
        else:
            merged_hom_data = pd.concat([merged_hom_data, hom_in_data_i], )

    merged_hom_data.replace(to_replace=[None], value=np.nan, inplace=True)
    return merged_hom_data


def get_celex_transcription(hom_in_data_df, celex_phonology_dict):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param celex_phonology_dict (pd.DataFrame): The dataframe with celex information for english words
    :return (pd.DataFrame): merged dataframe containing additional data:
    - disc
    - clx
    - disc_no_bound (no boundaries)
    - clx_no_bound (no boundaries)
    """
    return hom_in_data_df.merge(celex_phonology_dict[["word", "disc", "clx", "disc_no_bound", "clx_no_bound"]],
                    how = "left",
                    left_on=["word", "celexPhon"], #the celexPhon turned out to the disc encoding without boundaries
                    right_on=["word","disc_no_bound"])


def merge_m_scores_df_to_homophone_data(hom_in_data_df, m_scores_df):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param m_scores_df: The dataframe containing the m_scores for each homophone
    :return (pd.DataFrame): merged dataframe containing additional data
    - m_score : based on Berndt et al. 1987
    """
    return hom_in_data_df.merge(m_scores_df, on="word")


def merge_celex_syl_counts_df_to_homophone_data(hom_in_data_df, celex_df):
    """
    :param hom_in_data_df (pd.DataFrame): The dataframe containing homophones
    :param celex_df: he dataframe containing syllable counts from CELEX
    :return (pd.DataFrame): merged dataframe containing additional data
    """
    celex_subset_df = celex_df.drop_duplicates("Word")[["Word", "SylCnt"]].rename(columns={"Word": "word"})
    return hom_in_data_df.merge(celex_subset_df, on="word",how="left")
