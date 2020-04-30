import seg_files
import os
#filepath = 'Data/seg_files/2016-12-17_1330_US_KCET_Asia_Insight.seg'

FILE_EXT = '.seg'
LEN_FILE_EXT = len(FILE_EXT)

def get_seg_file_pos_info(filepath):
    file = os.path.basename(filepath)[:-LEN_FILE_EXT]
    lang, header, credit, body = seg_files.read_seg(filepath)
    pos1 = seg_files.get_pos1(body)
    pos1["prev"] = pos1["word"].shift(1)
    pos1["next"] = pos1["word"].shift(-1)
    pos1["end_of_sentence"] = pos1["next"].isin([".", "!", "?"])
    pos1["start_of_sentence"] = pos1["prev"].isin([".", "!", "?"])
    pos1["preceding_marker"] = pos1["prev"].isin([",", ".", "?", "!", ":", ";"])
    pos1["subsequent_marker"] = pos1["next"].isin([",", ".", "?", "!", ":", ";"])

    prev_words = pos1[~pos1.word.isin([",", ".", "?", "!", ":", ";"])]["word"].shift(1)
    next_words = pos1[~pos1.word.isin([",", ".", "?", "!", ":", ";"])]["word"].shift(-1)

    pos1.loc[~pos1.word.isin([",", ".", "?", "!", ":", ";"]), "prev_word"] = prev_words
    pos1.loc[~pos1.word.isin([",", ".", "?", "!", ":", ";"]), "next_word"] = next_words
    pos1["source_file"] = file
    #pos1.drop(columns=['prev', 'next'], inplace=True)

    return pos1
