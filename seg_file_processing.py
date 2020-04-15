import seg_files

#filepath = 'Data/seg_files/2016-12-17_1330_US_KCET_Asia_Insight.seg'


def get_seg_file_pos_info(filepath):
    lang, header, credit, body = seg_files.read_seg(filepath)
    pos1 = seg_files.get_pos1(body)
    pos1["next_word"] = pos1["word"].shift(-1)
    pos1["end_of_sentence"] = pos1["next_word"].isin([".", "!", "?"])
    pos1["prev_to_marker"] = pos1["next_word"].isin([",", ".", "?", "!", ":", ";"])

    pos1.drop(columns=['next_word'])
    return pos1
