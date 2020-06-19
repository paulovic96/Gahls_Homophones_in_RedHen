import pandas as pd
import numpy as np
import eaf_file_processing
import mp4_file_processing
import seg_file_processing
import gentle_file_processing
import tqdm
from time import sleep
import sys

FILE_BASE = '/mnt/Restricted/Corpora/RedHen' # 'Data'

EAF_FILE_EXT = ".eaf.gz"
VIDEO_FILE_EXT = ".mp4"
SEG_FILE_EXT = ".seg"
GENTLE_FILE_EXT = ".gentleoutput_v2.json.gz"


FILE_DESCRIPTIONS_TO_EXT = {"video": VIDEO_FILE_EXT, "eaf":EAF_FILE_EXT, "seg":SEG_FILE_EXT, "gentle":GENTLE_FILE_EXT}

def get_file_path(source_file, is_gentle_file=False):
    """
    :param source_file (str): file name
    :param is_gentle_file (bool): for a gentle file we have a different file structure
    :return file_path (str): corresponding file path according to the Year/Year-Month/Year-Month_Day/ folder structure
    """
    year,month,day = source_file.split("_")[0].split("-")
    if is_gentle_file:
        file_path = "%s_gentle_v2/%s-%s/%s-%s-%s/" % (year, year, month, year,month,day)
    else:
        file_path = "%s/%s-%s/%s-%s-%s/" % (year, year, month, year,month,day)
    file_path += source_file
    return file_path


def word_preprocessing(word):
    word = word.lower()
    return word

def calculate_duration(start,end):
    duration = end - start
    return duration

def calculate_frequency_by_column(df,column):
    """
    :param df (pd.DataFrame):
    :param column (str): a column in df
    :return: the frequency of each unique value in the provided column
    """
    return df.groupby([column])[column].transform("count")


def is_preceding_or_subsequent_to_pause(df,source_c = 'source_file', word_c = "word", pause_indicator = "<non-speech>"):
    """
    :param df (pd.DataFrame):
    :param source_c (str): name of column containing the source file
    :param word_c (str): name of column containing the words of interest
    :param pause_indicator (str): the string used to indicate a pause
    :return preceding_pause (pd.Series): series indicating whether a word has a preceding pause
    :return subsequent_pause (pd.Series): series indicating whether a word has a subsquent pause
    """
    pause_df = df.copy()
    pause_df["prev_word"] = pause_df.groupby([source_c])[word_c].shift(1)
    pause_df["next_word"] = pause_df.groupby([source_c])[word_c].shift(-1)

    pause_df["preceding_pause"] = pause_df["prev_word"] == pause_indicator
    pause_df["subsequent_pause"] = pause_df["next_word"] == pause_indicator

    return pause_df["preceding_pause"], pause_df["subsequent_pause"]


def read_dataframe(filename, remove_pauses=True, remove_errors=True, preprocessing=True, drop_error_columns=False):
    """
    :param filename (str): the filename under which the pickled dataset is stored
    :param remove_pauses (bool):
    :param remove_errors (bool):
    :param preprocessing (bool):
    :param drop_error_columns (bool):
    :return df (pd.DataFrame): the dataframe after having applied the preprocessing containing:
        - extracting pause information
        - convert strings to lower case
        - calculate duration
        - calculate word-frequency
        - extract context information of preceding and subsequent words and their frequencies
        - calculate length in letter
        - calculate contextual predictability based on prev and next words frequencies

    """
    print(f'read dataframe from {filename}')
    df = pd.read_pickle(filename)
    df.sort_values(by = ["source_file","start"], inplace = True)

    if remove_pauses:
        if preprocessing:
            print("Preprocessing: extract pause information...") # extract paus information before removing them
            df["preceding_pause"], df["subsequent_pause"] = is_preceding_or_subsequent_to_pause(df)
        print("Remove pauses from data!")
        df = df[df.word != "<non-speech>"]
    if remove_errors:
        df = df[df.mp4_error == 'no-error']
        df = df[df.aac_error == 'no-error']
        df = df[df.aac2wav_error == 'no-error']
        df = df[df.eafgz_error == 'no-error']
        df = df[df.seg_error == 'no-error']

    df = df.set_index(pd.RangeIndex(df.shape[0]))

    if preprocessing:
        print("Preprocessing: apply word preprocessing...")
        df.word = df.word.str.lower()#df.word.apply(word_preprocessing)
        print("Preprocessing: calculate word duration...")
        df.duration = df.end - df.start

        print("Preprocessing: calculate word frequency...")
        df["word_frequency"] = calculate_frequency_by_column(df,column="word")

        print("Preprocessing: extract context information...")
        if remove_pauses:
            df["prev_word"] = df.groupby(["source_file"])["word"].shift(1)
            df["prev_word_frequency"] = df.groupby(["source_file"])["word_frequency"].shift(1)
            df["next_word"] = df.groupby(["source_file"])["word"].shift(-1)
            df["next_word_frequency"] = df.groupby(["source_file"])["word_frequency"].shift(-1)

        else:
            prev_words = df[df.word != "<non-speech>"].groupby(["source_file"])["word"].shift(1)
            prev_word_frequencies = df[df.word != "<non-speech>"].groupby(["source_file"])["word_frequency"].shift(1)
            next_words = df[df.word != "<non-speech>"].groupby(["source_file"])["word"].shift(-1)
            next_word_frequencies = df[df.word != "<non-speech>"].groupby(["source_file"])["word_frequency"].shift(-1)

            df.loc[df.word != "<non-speech>", "prev_word"] = prev_words
            df.loc[df.word != "<non-speech>", "prev_word_frequency"] = prev_word_frequencies
            df.loc[df.word != "<non-speech>", "next_word"] = next_words
            df.loc[df.word != "<non-speech>", "next_word_frequency"] = next_word_frequencies

            print("Preprocessing: extract pause information...")
            preceding_pauses, subsequent_pauses = is_preceding_or_subsequent_to_pause(df)
            df.loc[df.word != "<non-speech>", "preceding_pause"] = preceding_pauses
            df.loc[df.word != "<non-speech>", "subsequent_pause"] = subsequent_pauses

        print("Preprocessing: calculate length in letter...")
        df["length_in_letter"] = df["word"].apply(lambda word: len(list(word)))

        print("Preprocessing: calculate contextual predictability...")
        df["prev_word_string"] = df['prev_word'] + "-" + df['word']
        df["next_word_string"] = df['word'] + "-" + df['next_word']

        df["prev_word_string_frequency"] = calculate_frequency_by_column(df, "prev_word_string")
        df["next_word_string_frequency"] = calculate_frequency_by_column(df, "next_word_string")

        df["cond_pred_prev"] = df["prev_word_string_frequency"] / df['prev_word_frequency']
        df["cond_pred_next"] = df["next_word_string_frequency"] / df['next_word_frequency']

    if drop_error_columns:
        df = df.drop(columns=['mp4_error', 'aac_error', 'aac2wav_error', 'eafgz_error'])
    print(df.shape, df.index)
    return df


def read_and_extract_homophones(hom_filename, data):
    """
    :param hom_filename (str): the filename under which the homophone dataframe is stored
    :param data (pd.DataFrame): the dataframe from which you want to extract the homophones
    :return homophones_in_data (pd.DataFrame): the dataframe reduced to only contain homophones
    - pronunciation frequency added
    - is_max factor introduced to indicate the more frequent homophone if both pairs are available otherwise always 1
    - has_pair indicating whether the homophone is part of a pair
    :return gahls_homophones: the loaded homophone dataframe stored under hom_filename
    :return gahls_homophones_missing_in_data: the subset of gahls_homophones df which contains the homophones not present in the data
    """
    print(f'read Gahls Homophone data from {hom_filename}')
    gahls_homophones = pd.read_csv(hom_filename, index_col="Unnamed: 0")


    homophones_present_in_data = gahls_homophones[gahls_homophones["spell"].isin(data["word"])]["spell"]
    homophones_missing_in_data = gahls_homophones[~gahls_homophones["spell"].isin(data["word"])]["spell"]
    gahls_homophones_in_data = gahls_homophones[gahls_homophones["spell"].isin(homophones_present_in_data)].copy()
    gahls_homophones_missing_in_data = gahls_homophones[gahls_homophones["spell"].isin(homophones_missing_in_data)].copy()


    homophones_in_data = data[data["word"].isin(gahls_homophones_in_data["spell"])].copy().drop_duplicates()

    pronunciation_count = pd.value_counts(gahls_homophones_in_data["pron"])
    homophone_pairs = pronunciation_count[pronunciation_count == 2]
    gahls_homophones_in_data_pairs = gahls_homophones_in_data[gahls_homophones_in_data["pron"].isin(homophone_pairs.index)]
    homophones_in_data["has_pair"] = data["word"].isin(gahls_homophones_in_data_pairs["spell"])

    # homophone_pairs_in_data = data[data["word"].isin(gahls_homophones_in_data_pairs["spell"])].copy()
    # homophone_pairs_in_data = homophone_pairs_in_data.drop_duplicates()

    print("%d out of %d homophones found in Data:" % (len(np.unique(homophones_in_data.word)), len(np.unique(gahls_homophones.spell))))
    pairs = homophones_in_data.groupby("word").first().has_pair
    print("Homophone Pairs found in Data:", int(np.sum(pairs)/2))
    print("Homophones without Pair: ", list(pairs.iloc[np.where(pairs == False)[0]].index))
    print("Missing homophones:", np.unique(gahls_homophones_missing_in_data.spell))


    gahls_homophones_in_data.rename(columns={'spell':'word'}, inplace=True)
    gahls_homophones_missing_in_data.rename(columns={'spell':'word'}, inplace=True)

    #homophone_pairs_in_data = homophone_pairs_in_data.merge(gahls_homophones_in_data[["word", "pron"]], on="word")
    #homophone_pairs_in_data["pron_frequency"] = calculate_frequency_by_column(homophone_pairs_in_data,column="pron")

    #max_idx = homophone_pairs_in_data.groupby(['pron'])['pron_frequency'].transform(max).copy() == homophone_pairs_in_data["pron_frequency"]
    #homophone_pairs_in_data["is_max"] = np.asarray(max_idx,dtype = np.int)


    homophones_in_data = homophones_in_data.merge(gahls_homophones_in_data[["word", "pron", "celexPhon"]], on="word")
    homophones_in_data["pron_frequency"] = calculate_frequency_by_column(homophones_in_data,column="pron")

    max_idx = homophones_in_data.groupby(['pron'])['pron_frequency'].transform(max).copy() == homophones_in_data["pron_frequency"]
    homophones_in_data["is_max"] = np.asarray(max_idx,dtype = np.int)


    return homophones_in_data, gahls_homophones, gahls_homophones_missing_in_data




def get_additional_data_from_files(df, file_description): # file description one of ["video", "eaf", "seg", "gentle"]
    """
    Helper function to call the right functions to extract and preprocess the additional data from video,eaf,seg and gentle files
    :param df (pd.DataFrame): The dataframe for which we want to get additional data from different files
    :param file_description (str): Description of files from which we want to extract the additional data
    :return: file_df (pd.DataFrame): A dataframe containing the additional data extract from the corresponding files
    """
    if file_description == "gentle":
        file_folder = FILE_BASE + "/gentle/"
        is_gentle_file = True
    else:
        file_folder = FILE_BASE + "/original/"
        is_gentle_file = False

    file_df = None

    if file_description not in list(FILE_DESCRIPTIONS_TO_EXT.keys()):
        print("Unknown file description! Don't know what to do with %s files..." % file_description)
        return None

    else:
        print("Load and extract information from %s files..." % file_description)
        #pbar = tqdm.tqdm(total = len(np.unique(df["source_file"])),desc='Files', position=0,leave=True,file=sys.stdout)
        #file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}',leave=True,file=sys.stdout)
        for file in np.unique(df["source_file"]):
            filepath = file_folder + get_file_path(file,is_gentle_file=is_gentle_file) + FILE_DESCRIPTIONS_TO_EXT[file_description]

            if file_description == "video":
                file_i_df = mp4_file_processing.get_word_video_snippet_size(df, filepath)
            elif file_description == "eaf":
                speech_annotation_eaf_data, gesture_eaf_data = eaf_file_processing.read_eaf(filepath)
                file_i_df = eaf_file_processing.map_gestures_to_annotation(speech_annotation_eaf_data, gesture_eaf_data, remove_pauses=False)
                file_i_df = eaf_file_processing.binary_encode_gestures(file_i_df, gesture_column="gesture")

            elif file_description == "seg":
                file_i_df = seg_file_processing.get_seg_file_pos_info(filepath)

            elif file_description == "gentle":
                file_i_df = gentle_file_processing.get_gentle_file_transcripts(filepath)
                
            else:
                print("Unknown file format!!!")
                return

            if file_df is None:
                file_df = file_i_df
            else:
                file_df = pd.concat([file_df, file_i_df], ignore_index=True)

            #file_log.set_description_str(f'Processed file: {file}')
            #pbar.update(1)
            #sleep(0.02)
        #file_log.close()
        #pbar.close()
        return file_df


