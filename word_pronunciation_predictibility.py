# pronunciation predictability

import pandas as pd
import numpy as np
from g2p_en import G2p # https://github.com/Kyubyong/g2p


CELEX_FILE = "/mnt/shared/corpora/Celex/english/epw/epw.cd" #"Data/english/epw/epw.cd"
BERNDT_CHARACTER_CODING_FILE = "/mnt/Restricted/Corpora/RedHen/phonetic_character_code_berndt1987.csv" #"Data/phonetic_character_code_berndt1987.csv"
BERNDT_CONDITIONAL_PROB_TABLE_FILE ="/mnt/Restricted/Corpora/RedHen/Conditional_Probabilities_for_Grapheme-to-Phoneme_Correspondences_Berndt1987.csv" #"Data/Conditional_Probabilities_for_Grapheme-to-Phoneme_Correspondences_Berndt1987.csv"


def get_ARPABET_phonetic_transcription(word_list):
    """
    :param word_list (list): List of words to encode with ARPABET phonetic transcription
    :return arpabet_word_list (list): List of lists of enocded phonemes
    """
    g2p = G2p()
    arpabet_word_list = []
    for word in word_list:
        transcription = g2p(word)
        arpabet_word_list.append(transcription)

    return arpabet_word_list


def get_english_phonology_from_celex(filename):
    """
    :param filename (str): file under which CELEX database is stored
    :return celex_phonology_dict (dict): dictionary with CELEY phonetic transcriptions
    """
    phonology_dict = {"word":[], "disc":[], "clx":[]}
    with open(filename) as f:
        for line in f:
            line = line.strip().split("\\")
            word = line[1] # the word
            phonology_dict["word"].append(word)
            disc = line[6] # pronunciation in DISC notation, hyphens to mark syllable boundaries, inverted comma for primary stress and double quote for secondary stress (PhonStrsDISC)
            phonology_dict["disc"].append(disc)
            clx = line[8] # pronunciation in CELEX notation, with brackets (PhonSylBCLX)
            phonology_dict["clx"].append(clx)

    celex_phonology_dict = pd.DataFrame.from_dict(phonology_dict).drop_duplicates()
    celex_phonology_dict["disc_no_bound"] = celex_phonology_dict["disc"].apply(
        lambda x: x.replace("'", "").replace("-", ""))
    celex_phonology_dict["clx_no_bound"] = celex_phonology_dict["clx"].apply(
        lambda x: x.replace("[", "").replace("]", ""))
    return celex_phonology_dict



def get_ARPABET_to_keyboard_phonetic_symbols_dict(berndt_character_code_df):
    """
    :param berndt_character_code_df (pd.DataFrame): The dataframe containing the Phoneme Equivalents of Berndt's keyboard compatible phonemic symbols and CELEX, DISC and ARPABET transcription
    :return berndt_arpabet_phon_dict (dict): A dictionary containing for each ARPABET symbol the corresponding keyboard compatible phonemic symbol
    """
    berndt_arpabet_phon_dict = {}
    for i, p in enumerate(berndt_character_code_df["g2p(ARPAbet)"]):
        if not p is np.nan:
            p = p.replace(" ", "").split(",")
            if len(p) > 1:
                for p_i in p:
                    berndt_arpabet_phon_dict[p_i] = berndt_character_code_df.keyboard_compatible_phonetic_symbol.iloc[i]
            else:
                berndt_arpabet_phon_dict[p[0]] = berndt_character_code_df.keyboard_compatible_phonetic_symbol.iloc[i]
    return berndt_arpabet_phon_dict


def get_keyboard_phonetic_symbols_for_ARPABET(arpabet_word, dictionary):
    """
    :param arpabet_word (list): list of ARPABET enocded phonemes
    :param dictionary (dict): dictionary containing for each ARPABET symbol Berndt's corresponding keyboard compatible phonemic symbol
    :return keyboard_encoding (list): list of keyboard compatible phonemic symbol for the ARPABET encoded word
    """
    skip_word = False
    keyboard_encoding = []
    for i, p in enumerate(arpabet_word):
        if skip_word == True:
            skip_word = False
        else:
            if p == 'AH0': # Berndt codes "ul","um","un" as a single sounds while arpabet uses AH0 followed by L/M/N which would correspond to e.g. "uh-" - "l" in Berndt's table
                if i < len(arpabet_word) - 1:
                    if arpabet_word[i + 1] == "L":
                        keyboard_encoding.append([dictionary[p + "-" + "L"], [dictionary[p], dictionary["L"]]]) # keep track of both possibilities e.g. "ul" and "uh-" "l"
                        skip_word = True
                    elif arpabet_word[i + 1] == "M":
                        keyboard_encoding.append([dictionary[p + "-" + "M"], [dictionary[p], dictionary["M"]]])
                        skip_word = True
                    elif arpabet_word[i + 1] == "N":
                        keyboard_encoding.append([dictionary[p + "-" + "N"], [dictionary[p], dictionary["N"]]])
                        skip_word = True
                    else:
                        keyboard_encoding.append(dictionary[p])
                else:
                    keyboard_encoding.append(dictionary[p])
            elif p == "K": # Berndt codes "ks" in e.g. Ox as a single sound while Arpabet uses "K" followed by "S"
                if i < len(arpabet_word) - 1:
                    if arpabet_word[i + 1] == "S":
                        keyboard_encoding.append([dictionary[p] + dictionary["S"], [dictionary[p], dictionary["S"]]])
                        skip_word = True
                    else:
                        keyboard_encoding.append(dictionary[p])
                else:
                    keyboard_encoding.append(dictionary[p])

            elif p == "Y": # Berndt codes "ju:" as a sinlge sound while Arpabet uses "Y" followed by one of ["UW0", "UW1", "AH0"]
                if i < len(arpabet_word) - 1:
                    if arpabet_word[i + 1] in ["UW0", "UW1", "AH0"]:
                        keyboard_encoding.append([dictionary[p + "-" + arpabet_word[i + 1]],
                                                  [dictionary[p], dictionary[arpabet_word[i + 1]]]])
                        skip_word = True
                    else:
                        keyboard_encoding.append(dictionary[p])
                else:
                    keyboard_encoding.append(dictionary[p])
            elif p == "AO1": # Berndt codes a "ou" followed by an "r" like in four (see condition prob table example) as "o" sound but Arpabet codes it as A01 which would correspond to an "aw" sound in Berndts coding
                if i < len(arpabet_word) - 1:
                    if arpabet_word[i + 1] == "R":
                        keyboard_encoding.append([dictionary[p], [dictionary["OW0"]]])
                    else:
                        keyboard_encoding.append(dictionary[p])
                else:
                    keyboard_encoding.append(dictionary[p])

            elif p == "IH1": # The "ea" in words like "dear" and "shear" are encoded by IH1 in Arpabet this maps to an "ih" sound in Berndt's table. However an "ih" sound in her conditional prob table does not occur for a AE grapheme string.
                # With respect to the cambridge dictionary is "hear" another example for the sound of "ea". However the "ea" in "hear" is encoded by IY1 which would correspond to a "ee" sound for which we have the mapping to "ea" in Berndt's table.
                keyboard_encoding.append([dictionary[p], [dictionary["IY1"]]])

            elif p in ["ER0", "ER1", "ER2"]: # Berndt codes the "e" in "er" like in baker (see condition prob table example) as "er" sound while Arpabet encodes the whole "er" by one of ["ER0", "ER1", "ER2"].
                # for words like fire, hire the re is encoded by "ER-" in Arpabet but includes referring to the literature a silent e thats why we take a simple "r" sound into account
                keyboard_encoding.append([dictionary[p], dictionary["R"], [dictionary[p], dictionary["R"]]])
            else:
                keyboard_encoding.append(dictionary[p])
    return keyboard_encoding


def get_keyboard_phonetic_symbols_to_grapheme_cond_prob_dict(conditional_prob_df):
    """
    :param conditional_prob_df (pd.DataFrame): The dataframe containing Berndt's Conditional Probabilities for Grapheme-to-Phoneme Correspondences
    :return phonem_graphem_prob_dict (dict): Dictionary with:   keys = keyboard compatible phonemic symbols
                                                                values = list of tuple of (1. Grapheme, 2. Prio Prob, 3. Cond Prob)
    """
    phonem_graphem_prob_dict = {}
    for phoneme in conditional_prob_df.Phoneme.unique():
        phonem_graphem_prob_dict[phoneme] = []

    for i, row in conditional_prob_df.iterrows():
        grapheme_prior_cond = (row["Grapheme"], row['Prior_Probability'], row["Conditional_Probability"])
        phonem_graphem_prob_dict[row["Phoneme"]].append(grapheme_prior_cond)

    return phonem_graphem_prob_dict


def get_grapheme_string_with_conditional_prob_for_keyboard_phonetics(word_phon_tuples, phonem_graphem_prob_dict):
    """
    :param word_phon_tuples (list): A list of tuple with 1. the actual words and 2. the phonetic transcription as list of keyboard compatible phonemic symbols
    :param phonem_graphem_prob_dict (dict): A Dictionary with:  keys = keyboard compatible phonemic symbols
                                                                values = list of tuple of (1. Grapheme, 2. Prior Prob, 3. Cond Prob)
    :return possible_grapheme_strings (list): A list of lists with possible grapheme mappings given the pronunciation of a word
    :return possible_prior_probs (list): A list of lists with the corresponding prior probs for the possible grapheme mappings
    :return possible_cond_probs (list): A list of lists with the corresponding cond probs for the possible grapheme mappings
    :return word_rests (list): A list of lists with the rest of the actual known word after concatenating the matching possible graphemes
    """
    possible_grapheme_strings = []  # list for each word
    possible_prior_probs = []  # list for each word
    possible_cond_probs = []  # list for each word
    word_rests = []  # list for each word

    for i, word_pron in enumerate(word_phon_tuples):
        word = word_pron[0]  # word string
        pron = word_pron[1]  # list of keyboard compatible phon characters
        #print(word, pron)
        if word[0] == 'h' and pron[0] != 'h':  # leading silent h
            possible_grapheme_strings_i = [
                ['$H']]  # for each word a list of possible lists with grapheme given pronunciation strings
            possible_prior_probs_i = [
                [0.0003]]  # for each word a list of possible lists with corresponding prio probabilites
            possible_cond_probs_i = [
                [1.000]]  # for each word a list of possible lists possible corresponding conditional probabilities
            word_rests_i = [word.upper()[
                            1:]]  # for each word a list of remaining word characters after having splitted it into a list of possible graphemes

        else:
            possible_grapheme_strings_i = [
                []]  # for each word a list of possible lists with grapheme given pronunciation strings
            possible_prior_probs_i = [
                []]  # for each word a list of possible lists with corresponding prio probabilites
            possible_cond_probs_i = [
                []]  # for each word a list of possible lists possible corresponding conditional probabilities
            word_rests_i = [
                word.upper()]  # for each word a list of remaining word characters after having splitted it into a list of possible graphemes

        for j, p in enumerate(pron):
            new_word_rests_i = []  # new rest of the word after looking at the current encoded syllable pronunciation
            new_possible_grapheme_strings_i = []  # new possible grapheme strings given the current syllable pronunciation
            new_possible_prior_probs_i = []
            new_possible_cond_probs_i = []
            if isinstance(p, list):
                for p_i in p:
                    if isinstance(p_i, list):
                        for p_ij in p_i:
                            for possible_grapheme in phonem_graphem_prob_dict[p_ij]:  # all possible corresponding graphemes
                                grapheme = possible_grapheme[0].split("-")  # account for silent e encoded by e.g. A-E
                                prior = possible_grapheme[1]  # prior prob
                                cond = possible_grapheme[2]  # cond prop
                                for k, word_rest in enumerate(
                                        word_rests_i):  # for each possible combination we get a different word rest e.g. APPLE can have [A,P], [A,PP], [A-E,P], [A-E,PP] --> ["PLE", LE, PL, L]
                                    if len(grapheme) > 1:  # silent E
                                        # if word_rest.startswith(grapheme[0]) and word_rest.endswith(grapheme[1]): # check whether the grapheme fits to the rest of the word
                                        if word_rest.startswith(grapheme[0]) and grapheme[1] in word_rest[
                                                                                                len(grapheme[0]):]:
                                            new_possible_grapheme_strings_i.append(possible_grapheme_strings_i[k] + [
                                                possible_grapheme[
                                                    0]])  # add the grapheme to the grapheme list which corresponds to the word ret we are currently looking at
                                            new_possible_prior_probs_i.append(possible_prior_probs_i[k] + [prior])
                                            new_possible_cond_probs_i.append(possible_cond_probs_i[k] + [cond])
                                            # new_word_rests_i.append(word_rest[len(grapheme[0]):-1]) # new word_rests
                                            new_word_rests_i.append(
                                                word_rest[len(grapheme[0]):].replace(grapheme[1], "", 1))
                                    else:
                                        if word_rest.startswith(grapheme[0]):  # no silent E
                                            new_possible_grapheme_strings_i.append(
                                                possible_grapheme_strings_i[k] + [possible_grapheme[0]])
                                            new_possible_prior_probs_i.append(possible_prior_probs_i[k] + [prior])
                                            new_possible_cond_probs_i.append(possible_cond_probs_i[k] + [cond])
                                            new_word_rests_i.append(word_rest[len(grapheme[0]):])

                            word_rests_i = new_word_rests_i  # update word rests
                            possible_grapheme_strings_i = new_possible_grapheme_strings_i  # update possible grapheme strings
                            possible_prior_probs_i = new_possible_prior_probs_i
                            possible_cond_probs_i = new_possible_cond_probs_i


                    else:  # proceed like normal but without updating the words_rests
                        for possible_grapheme in phonem_graphem_prob_dict[p_i]:  # all possible corresponding graphemes
                            grapheme = possible_grapheme[0].split("-")  # account for silent e encoded by e.g. A-E
                            prior = possible_grapheme[1]  # prior prob
                            cond = possible_grapheme[2]  # cond prop

                            for k, word_rest in enumerate(
                                    word_rests_i):  # for each possible combination we get a different word rest e.g. APPLE can have [A,P], [A,PP], [A-E,P], [A-E,PP] --> ["PLE", LE, PL, L]
                                if len(grapheme) > 1:  # silent E
                                    # if word_rest.startswith(grapheme[0]) and word_rest.endswith(grapheme[1]): # check whether the grapheme fits to the rest of the word
                                    if word_rest.startswith(grapheme[0]) and grapheme[1] in word_rest[
                                                                                            len(grapheme[0]):]:
                                        new_possible_grapheme_strings_i.append(possible_grapheme_strings_i[k] + [
                                            possible_grapheme[
                                                0]])  # add the grapheme to the grapheme list which corresponds to the word ret we are currently looking at
                                        new_possible_prior_probs_i.append(possible_prior_probs_i[k] + [prior])
                                        new_possible_cond_probs_i.append(possible_cond_probs_i[k] + [cond])
                                        # new_word_rests_i.append(word_rest[len(grapheme[0]):-1]) # new word_rests
                                        new_word_rests_i.append(
                                            word_rest[len(grapheme[0]):].replace(grapheme[1], "", 1))
                                else:
                                    if word_rest.startswith(grapheme[0]):  # no silent E
                                        new_possible_grapheme_strings_i.append(
                                            possible_grapheme_strings_i[k] + [possible_grapheme[0]])
                                        new_possible_prior_probs_i.append(possible_prior_probs_i[k] + [prior])
                                        new_possible_cond_probs_i.append(possible_cond_probs_i[k] + [cond])
                                        new_word_rests_i.append(word_rest[len(grapheme[0]):])
            else:
                for possible_grapheme in phonem_graphem_prob_dict[p]:  # all possible corresponding graphemes
                    grapheme = possible_grapheme[0].split("-")  # account for silent e encoded by e.g. A-E
                    prior = possible_grapheme[1]  # prior prob
                    cond = possible_grapheme[2]  # cond prop
                    for k, word_rest in enumerate(
                            word_rests_i):  # for each possible combination we get a different word rest e.g. APPLE can have [A,P], [A,PP], [A-E,P], [A-E,PP] --> ["PLE", LE, PL, L]
                        if len(grapheme) > 1:  # silent E
                            # if word_rest.startswith(grapheme[0]) and word_rest.endswith(grapheme[1]): # check whether the grapheme fits to the rest of the word
                            if word_rest.startswith(grapheme[0]) and grapheme[1] in word_rest[len(grapheme[0]):]:
                                new_possible_grapheme_strings_i.append(possible_grapheme_strings_i[k] + [
                                    possible_grapheme[
                                        0]])  # add the grapheme to the grapheme list which corresponds to the word ret we are currently looking at
                                new_possible_prior_probs_i.append(possible_prior_probs_i[k] + [prior])
                                new_possible_cond_probs_i.append(possible_cond_probs_i[k] + [cond])
                                # new_word_rests_i.append(word_rest[len(grapheme[0]):-1]) # new word_rests
                                new_word_rests_i.append(word_rest[len(grapheme[0]):].replace(grapheme[1], "", 1))

                        else:
                            if word_rest.startswith(grapheme[0]):  # no silent E
                                new_possible_grapheme_strings_i.append(
                                    possible_grapheme_strings_i[k] + [possible_grapheme[0]])
                                new_possible_prior_probs_i.append(possible_prior_probs_i[k] + [prior])
                                new_possible_cond_probs_i.append(possible_cond_probs_i[k] + [cond])
                                new_word_rests_i.append(word_rest[len(grapheme[0]):])

                word_rests_i = new_word_rests_i  # update word rests
                possible_grapheme_strings_i = new_possible_grapheme_strings_i  # update possible grapheme strings
                possible_prior_probs_i = new_possible_prior_probs_i
                possible_cond_probs_i = new_possible_cond_probs_i

            # print(possible_grapheme_strings_i,word_rests_i)
        possible_grapheme_strings.append(possible_grapheme_strings_i)
        possible_prior_probs.append(possible_prior_probs_i)
        possible_cond_probs.append(possible_cond_probs_i)
        word_rests.append(word_rests_i)

    return possible_grapheme_strings, possible_prior_probs, possible_cond_probs, word_rests


def get_valid_grapheme_strings(word_phonem_tuples, possible_grapheme_strings, word_rests, possible_prior_probs,
                               possible_cond_probs):
    """
    :param word_phonem_tuples (list): A list of tuple with 1. the actual words and 2. the phonetic transcription as list of keyboard compatible phonemic symbols
    :param possible_grapheme_strings (list): A list of lists with possible grapheme mappings given the pronunciation of a word
    :param word_rests (list): A list of lists with the rest of the actual known word after concatenating the matching possible graphemes
    :param possible_prior_probs (list): A list of lists with the corresponding prior probs for the possible grapheme mappings
    :param possible_cond_probs (list): A list of lists with the corresponding cond probs for the possible grapheme mappings

    :return valid_word_rests (list): A List of hopefully empty strings
    :return valid_grapheme_strings (list): A list of lists with possible grapheme mappings given the pronunciation of a word and without any word rest remaining
    :return valid_prior_probs (list): A list of lists with the corresponding prior probs for the valid grapheme mappings
    :return valid_cond_probs (list): A list of lists with the corresponding cond probs for the valid grapheme mappings
    """
    valid_word_rests = []
    valid_grapheme_strings = []
    valid_prior_probs = []
    valid_cond_probs = []

    for i, word_tuple in enumerate(word_phonem_tuples):
        word = word_tuple[0]
        phon = word_tuple[1]

        word_rests_i = np.asarray(word_rests[i])
        grapheme_strings_i = np.asarray(possible_grapheme_strings[i])
        prio_probs_i = np.asarray(possible_prior_probs[i])
        cond_probs_i = np.asarray(possible_cond_probs[i])

        if len(word_rests_i) == 0:
            not_empty_word_rests_idx = []
        else:
            not_empty_word_rests_idx = np.where(word_rests_i != '')[0]

        if len(not_empty_word_rests_idx) > 0:
            valid_word_rests.append(np.delete(word_rests_i, not_empty_word_rests_idx, axis=0))
            valid_grapheme_strings.append(np.delete(grapheme_strings_i, not_empty_word_rests_idx, axis=0))
            valid_prior_probs.append(np.delete(prio_probs_i, not_empty_word_rests_idx, axis=0))
            valid_cond_probs.append(np.delete(cond_probs_i, not_empty_word_rests_idx, axis=0))
        else:
            valid_word_rests.append(word_rests_i)
            valid_grapheme_strings.append(grapheme_strings_i)
            valid_prior_probs.append(prio_probs_i)
            valid_cond_probs.append(cond_probs_i)

    return valid_word_rests, valid_grapheme_strings, valid_prior_probs, valid_cond_probs


def get_max_cond_prob_for_grapheme(conditional_probs_df):
    """
    :param conditional_prob_df (pd.DataFrame): The dataframe containing Berndt's Conditional Probabilities for Grapheme-to-Phoneme Correspondences
    :return max_cond_prob_for_grapheme (dict): Dictionary containing for each grapheme the probability of the most frequent correspondence for a grapheme
    """
    max_cond_prob_for_grapheme = conditional_probs_df.groupby("Grapheme").max()["Conditional_Probability"].to_dict()
    return max_cond_prob_for_grapheme



def calculate_m_scores(valid_grapheme_strings,valid_cond_probs, max_cond_prob_for_grapheme):
    """
    :param valid_grapheme_strings (list): A list of lists with possible grapheme mappings given the pronunciation of a word and without any word rest remaining
    :param valid_cond_probs (list): A list of lists with the corresponding cond probs for the valid grapheme mappings
    :param max_cond_prob_for_grapheme (dict): Dictionary containing for each grapheme the probability of the most frequent correspondence for a grapheme
    :return m_scores (list): A list of list of m-scores for each valid grapheme string
    """
    m_scores = []
    for i, grapheme_strings_i in enumerate(valid_grapheme_strings):
        m_score_i = []
        for j, possible_string_ij in enumerate(grapheme_strings_i):
            cond_prob = np.asarray(valid_cond_probs[i][j])
            most_prob = np.asarray([max_cond_prob_for_grapheme[g] for g in possible_string_ij])

            m_score_i.append(np.mean(cond_prob/most_prob))

        m_scores.append(m_score_i)
    return m_scores

def get_m_score_df(word_phonem_tuples, valid_grapheme_strings,valid_cond_probs,max_cond_prob_for_grapheme):
    """
    :param word_phonem_tuples (list): A list of tuple with 1. the actual words and 2. the phonetic transcription as list of keyboard compatible phonemic symbols
    :param valid_grapheme_strings (list): A list of lists with possible grapheme mappings given the pronunciation of a word and without any word rest remaining
    :param valid_cond_probs (list): A list of lists with the corresponding cond probs for the valid grapheme mappings
    :param max_cond_prob_for_grapheme (dict): Dictionary containing for each grapheme the probability of the most frequent correspondence for a grapheme
    :return (pd.DataFrame): A datafrane containing the average m-score for each word
    """
    m_scores = calculate_m_scores(valid_grapheme_strings,valid_cond_probs, max_cond_prob_for_grapheme)
    m_score_dict = {}
    for i, word_tuple in enumerate(word_phonem_tuples):
        word = word_tuple[0]
        phon = word_tuple[1]

        if len(m_scores[i]) > 0:
            m_score_dict[word] = np.mean(m_scores[i])
        else:
            m_score_dict[word] = np.nan

    return pd.DataFrame({"word":list(m_score_dict.keys()),"m_score" : list(m_score_dict.values())})


