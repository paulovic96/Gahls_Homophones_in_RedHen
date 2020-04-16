import seg_files
import os
#filepath = 'Data/seg_files/2016-12-17_1330_US_KCET_Asia_Insight.seg'

FILE_EXT = '.seg'
LEN_FILE_EXT = len(FILE_EXT)

def get_seg_file_pos_info(filepath):
    file = os.path.basename(filepath)[:-LEN_FILE_EXT]
    lang, header, credit, body = seg_files.read_seg(filepath)
    pos1 = seg_files.get_pos1(body)
    pos1["next_word"] = pos1["word"].shift(-1)
    pos1["end_of_sentence"] = pos1["next_word"].isin([".", "!", "?"])
    pos1["prev_to_marker"] = pos1["next_word"].isin([",", ".", "?", "!", ":", ";"])
    pos1["source_file"] = file
    pos1 = pos1.drop(columns=['next_word'])
    return pos1
