# format: http://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/red-hen-data-format
# HEADER: starts with TOP| and ends with LBT|
# CREDIT BLOCK: primary tag | date and time | source program | source person
# MAIN BODY: start time | end time | primary tag | content


# primary tags

    # FRM_01 -- linguistic frames, from FrameNet 1.5 via Semafor 3.0-alpha4
    # GES_02 -- 160 timeline gestures tagged manually by the Spanish gesture research group
    # GES_03 -- gestures tagged manually with ELAN
    # NER_03 -- named entities, using the Stanford NER tagger 3.4
    # POS_01 -- English parts of speech with dependencies, using MBSP 1.4
    # POS_02 -- English parts of speech, using the Stanford POS tagger 3.4
    # POS_03 -- German parts of speech, using the parser in Pattern.de
    # POS_04 -- French parts of speech, using the parser in Pattern.fr
    # POS_05 -- Spanish parts of speech, using the parser in pattern.es
    # SEG      -- Story boundaries by Weixin Li, UCLA
    # SEG_00 -- Commercial boundaries, using caption type information from CCExtractor 0.74
    # SEG_01 -- Commercial Detection by Weixin Li
    # SEG_02 -- Story boundaries by Rongda Zhu, UIUC
    # SMT_01 -- Sentiment detection, using Pattern 2.6
    # SMT_02 -- Sentiment detection, using SentiWordNet 3.0
    # DEU_01 -- German to English machine translation


import re
import pandas as pd


def read_seg(filepath):
    with open(filepath, 'r') as file:
        content = file.readlines()
    for ind, line in enumerate(content):
        line = line.strip().split("|")
        if line[0] == "LAN":
            lang = line[1]
        else:
            lang = None
        if line[0] == "LBT":
            end_header = ind
        if re.fullmatch('^[A-Z]{3}_\d{2}', line[0]):
            end_credit = ind
        if line[0] == 'END':
            end = ind
    header = content[:(end_header+1)]
    credit = content[(end_header+1):(end_credit+1)]
    body = content[(end_credit+1):end]
    return lang, header, credit, body

def get_pos1(body):
    """
    :param body: The body of the seg file
    :return (pd.DataFrame): Dataframe containin pos1 (English parts of speech with dependencies, using MBSP 1.4) information:
    - word
    - pos : part of speech
    - rel1: relation1
    - rel2: relation2
    - lemma: lemma
    """
    pos1 = {"word":[], "pos":[], "rel1":[], "rel2":[], "lemma":[]}
    for line in body:
        line = line.strip().split("|")
        if line[2] == "POS_01":
            words_with_tags = line[3:]
            words_with_tags_splitted = []
            for x in words_with_tags:
                if "/" in x:
                    words_with_tags_splitted.append(x.split("/"))
                else:
                    words_with_tags_splitted.append([float('nan'),float('nan'),float('nan'),float('nan'),float('nan')])
            #words_with_tags = [x.split("/") for x in words_with_tags]
            words, pos, rel1, rel2, lemma = map(list, zip(*words_with_tags_splitted))
            
            pos1["word"] += words
            pos1["pos"] += pos
            pos1["rel1"] += rel1
            pos1["rel2"] += rel2
            pos1["lemma"] += lemma

    return pd.DataFrame.from_dict(pos1)


def get_ges2(body):
    raise NotImplenetedError("to be implemented")

def get_ges3(body):
    raise NotImplenetedError("to be implemented")
