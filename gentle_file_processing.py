import gzip
import json
import pandas as pd
import os

FILE_EXT = '.gentleoutput_v2.json.gz'
LEN_FILE_EXT = len(FILE_EXT)

filepath = "Data/gentle_files/2016-12-17_1330_US_KCET_Asia_Insight.gentleoutput_v2.json.gz"

def get_gentle_file_transcripts(filepath):
    file = os.path.basename(filepath)[:-LEN_FILE_EXT]
    with gzip.open(filepath, "rb") as infile:
        json_file = json.load(infile)
        text = json_file['transcript'].split("\n")

    gentle_df = pd.DataFrame(text, columns=["word"])
    gentle_df["prev"] = gentle_df["word"].shift(1)
    gentle_df["next"] = gentle_df["word"].shift(-1)
    gentle_df["end_of_sentence"] = gentle_df["next"].isin([".", "!", "?"])
    gentle_df["start_of_sentence"] = gentle_df["prev"].isin([".", "!", "?"])
    gentle_df["preceding_marker"] = gentle_df["prev"].isin([",", ".", "?", "!", ":", ";"])
    gentle_df["subsequent_marker"] = gentle_df["next"].isin([",", ".", "?", "!", ":", ";"])
    gentle_df["source_file"] = file

    prev_words = gentle_df[~gentle_df.word.isin([",", ".", "?", "!", ":", ";"])]["word"].shift(1)
    next_words = gentle_df[~gentle_df.word.isin([",", ".", "?", "!", ":", ";"])]["word"].shift(-1)

    gentle_df.loc[~gentle_df.word.isin([",", ".", "?", "!", ":", ";"]), "prev_word"] = prev_words
    gentle_df.loc[~gentle_df.word.isin([",", ".", "?", "!", ":", ";"]), "next_word"] = next_words

    #gentle_df.drop(columns=['prev', 'next'], inplace=True)


    return gentle_df

