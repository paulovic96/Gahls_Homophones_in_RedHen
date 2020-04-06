import pandas as pd
import numpy as np

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

    pronunciation_count = pd.value_counts(gahls_homophones_in_data["pron"])
    homophone_pairs = pronunciation_count[pronunciation_count == 2]

    gahls_homophones_in_data_pairs = gahls_homophones_in_data[gahls_homophones_in_data["pron"].isin(homophone_pairs.index)]
    homophone_pairs_in_data = data[data["word"].isin(gahls_homophones_in_data_pairs["spell"])].copy()

    homophone_pairs_in_data = homophone_pairs_in_data.drop_duplicates()
    homophones_in_data = data[data["word"].isin(gahls_homophones_in_data["spell"])].copy().drop_duplicates()

    print("Homophone Pairs found in Data:", int(len(np.unique(homophone_pairs_in_data.word))/2))
    print("%d out of %d homophones found in Data:" % (len(np.unique(homophones_in_data.word)), len(np.unique(gahls_homophones.spell))))
    print("Missing homophones:", np.unique(gahls_homophones_missing_in_data.spell))


    gahls_homophones_in_data.rename(columns={'spell':'word'}, inplace=True)
    gahls_homophones_missing_in_data.rename(columns={'spell':'word'}, inplace=True)

    homophone_pairs_in_data = homophone_pairs_in_data.merge(gahls_homophones_in_data[["word", "pron"]], on="word")
    homophone_pairs_in_data["pron_frequency"] = calculate_frequency_by_column(homophone_pairs_in_data,column="pron")

    max_idx = homophone_pairs_in_data.groupby(['pron'])['pron_frequency'].transform(max).copy() == homophone_pairs_in_data["pron_frequency"]
    homophone_pairs_in_data["is_max"] = np.asarray(max_idx,dtype = np.int)


    homophones_in_data = homophones_in_data.merge(gahls_homophones_in_data[["word", "pron"]], on="word")
    homophones_in_data["pron_frequency"] = calculate_frequency_by_column(homophones_in_data,column="pron")


    return homophone_pairs_in_data, homophones_in_data, gahls_homophones, gahls_homophones_in_data, gahls_homophones_missing_in_data


def calculate_contextual_predictability(hom_data, word_column = "word", prev_word_column = "prev_word", next_word_column = "next_word", prev_word_freq_column = "prev_word_frequency", next_word_freq_column = "next_word_frequency" ):
    hom_data["prev_word_string"] = hom_data[prev_word_column] + "-" + hom_data[word_column]
    hom_data["next_word_string"] = hom_data[word_column] + "_" + hom_data[next_word_column]

    hom_data["prev_word_string_frequency"] = calculate_frequency_by_column(hom_data,"prev_word_string")
    hom_data["next_word_string_frequency"] = calculate_frequency_by_column(hom_data,"next_word_string")

    hom_data["cond_pred_prev"] = hom_data["prev_word_string_frequency"]/hom_data[prev_word_freq_column]
    hom_data["cond_pred_next"] = hom_data["next_word_string_frequency"]/hom_data[next_word_freq_column]

    return hom_data


def calculate_letter_length(df, word_column="word"):
    df["letter_length"] = df[word_column].apply(lambda word: len(list(word)))
    return df


#def
