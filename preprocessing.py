import pandas as pd
import numpy as np
import eaf_file_processing
import mp4_file_processing
import seg_file_processing
import gentle_file_processing


FILE_BASE = 'Data' #'/mnt/Restricted/Corpora/RedHen'

EAF_FILE_EXT = ".eaf.gz"
VIDEO_FILE_EXT = ".mp4"
SEG_FILE_EXT = ".seg"
GENTLE_FILE_EXT = ".gentleoutput_v2.json.gz"


FILE_DESCRIPTIONS_TO_EXT = {"video": VIDEO_FILE_EXT, "eaf":EAF_FILE_EXT, "seg":SEG_FILE_EXT, "gentle":GENTLE_FILE_EXT}

def get_file_path(source_file, is_gentle_file=False):
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
    return df.groupby([column])[column].transform("count")

def get_previous_or_next_word(df,shift,word_column= "word", source_column = "source_file", start_column = "start"):
    sorted_df = df.sort_values(by=[source_column, start_column])
    if shift == "prev":
        shifted_word_column = sorted_df.groupby([source_column])[word_column].shift(1)
    if shift == "next":
        shifted_word_column = sorted_df.groupby([source_column])[word_column].shift(-1)
    return shifted_word_column



def read_dataframe(filename, remove_pauses=True, remove_errors=True, preprocessing=True, drop_error_columns=False):
    print(f'read dataframe from {filename}')
    df = pd.read_pickle(filename)
    if remove_pauses:
        df = df[df.word != "<non-speech>"]
    if remove_errors:
        df = df[df.mp4_error == 'no-error']
        df = df[df.aac_error == 'no-error']
        df = df[df.aac2wav_error == 'no-error']
        df = df[df.eafgz_error == 'no-error']
        df = df[df.seg_error == 'no-error']
    df = df.set_index(pd.RangeIndex(df.shape[0]))
    if preprocessing:
        df.word = df.word.apply(word_preprocessing)
        df.duration = df.apply(lambda row: calculate_duration(row["start"], row['end']), axis=1)
        df["word_frequency"] = calculate_frequency_by_column(df,column="word")
        df["prev_word"] = get_previous_or_next_word(df,shift = "prev")
        df["prev_word_frequency"] = get_previous_or_next_word(df,shift="prev", word_column="word_frequency")
        df["next_word"] = get_previous_or_next_word(df,shift = "next")
        df["next_word_frequency"] = get_previous_or_next_word(df, shift="next", word_column="word_frequency")
        df["letter_length"] = df["word"].apply(lambda word: len(list(word)))

    if drop_error_columns:
        df = df.drop(columns=['mp4_error', 'aac_error', 'aac2wav_error', 'eafgz_error'])
    print(df.shape, df.index)
    return df


def read_and_extract_homophones(hom_filename, data):
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


def calculate_contextual_predictability_of_homophones(hom_data, word_column = "word", prev_word_column = "prev_word", next_word_column = "next_word", prev_word_freq_column = "prev_word_frequency", next_word_freq_column = "next_word_frequency" ):
    hom_data["prev_word_string"] = hom_data[prev_word_column] + "-" + hom_data[word_column]
    hom_data["next_word_string"] = hom_data[word_column] + "_" + hom_data[next_word_column]

    hom_data["prev_word_string_frequency"] = calculate_frequency_by_column(hom_data,"prev_word_string")
    hom_data["next_word_string_frequency"] = calculate_frequency_by_column(hom_data,"next_word_string")

    hom_data["cond_pred_prev"] = hom_data["prev_word_string_frequency"]/hom_data[prev_word_freq_column]
    hom_data["cond_pred_next"] = hom_data["next_word_string_frequency"]/hom_data[next_word_freq_column]

    return hom_data



def get_additional_data_from_files(df, file_description): # ["video", "eaf", "seg", "gentle"]
    if file_description == "gentle":
        file_folder = FILE_BASE + "/gentle/"
        is_gentle_file = True
    else:
        file_folder = FILE_BASE + "/original/"
        is_gentle_file = False

    file_df = None

    for file in np.unique(df["source_file"]):
        filepath = file_folder + get_file_path(file,is_gentle_file=is_gentle_file) + FILE_DESCRIPTIONS_TO_EXT[file_description]
        #print(filepath)
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
            print("Unknown file description! Don't know what to do with %s files..." % file_description)

        if file_df is None:
            file_df = file_i_df
        else:
            file_df = pd.concat([file_df, file_i_df], ignore_index=True)

    return file_df
