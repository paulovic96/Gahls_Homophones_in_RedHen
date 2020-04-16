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

    gentle_df = pd.DataFrame(text, columns=["gentle_transcription"])
    gentle_df["source_file"] = file
    return gentle_df

