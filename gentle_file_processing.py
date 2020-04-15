import gzip
import json



#filepath = "Data/gentle_files/2016-12-17_1330_US_KCET_Asia_Insight.gentleoutput_v2.json.gz"

def get_gentle_file_transcripts(filepath):
    with gzip.open(filepath, "rb") as infile:
        json_file = json.load(infile)
        text = json_file['transcript'].split("\n")
    return text
