import pandas as pd
import numpy as np
import torch
import os
import librosa
from pyannote.audio.utils.signal import Peak
import syllables


BASE = '/mnt/Restricted/Corpora/RedHen'
DATA_FOLDER = os.path.join(BASE, 'original')
DF_SOURCE_PATH = os.path.join(BASE, '2016_all_words_no_audio.pickle')
DF_HOMEOHONES_PATH = os.path.join(BASE, 'homophone_analysis_scripts/2016_all_words_no_audio_preprocessed.csv')
celex_dict_file = "/mnt/shared/corpora/Celex/english/epw/epw.cd"

STRETCHES_PATH = BASE + '/homophone_analysis_scripts'
#STRETCHES_PATH = '/mnt/shared/people/elnaz/homophones/10sec_stretch/'

NS = '<non-speech>'


def get_file_path(file_name, file_extension):
    base_name = os.path.basename(file_name)
    year, month, day = base_name[:10].split('-')
    file_path = f'{DATA_FOLDER}/{year}/{year}-{month}/{year}-{month}-{day}/{base_name}{file_extension}'
    return file_path

def get_prev_context(row):
    source_file, word, start, end = row[['source_file', 'word', 'start', 'end']]
    prev_context = df_source.loc[(df_source.source_file == source_file) &
                                 (start - df_source.start >= 0) & 
                                 (start - df_source.start <= 10) & 
                                 (start - df_source.end >= 0) &
                                 (start - df_source.end <= 10)]
    interruptions = prev_context[(prev_context.word == NS) &
                                 (prev_context.duration >= 0.4)] 
    if interruptions.empty:
        return prev_context[prev_context.word != NS]
    return prev_context[prev_context.start >= interruptions.iloc[-1].end]

def get_next_context(row):
    source_file, word, start, end = row[['source_file', 'word', 'start', 'end']]
    next_context = df_source.loc[(df_source.source_file == source_file) &
                                 (df_source.end - end >= 0) & 
                                 (df_source.end - end <= 10) & 
                                 (df_source.start - end >= 0) &
                                 (df_source.start - end <= 10)]
    interruptions = next_context[(next_context.word == NS) &
                                 (next_context.duration >= 0.4)] 
    if interruptions.empty:
        return next_context[next_context.word != NS]
    return next_context[next_context.end <= interruptions.iloc[0].start]


def get_context(source_file, word, start, end, df_source_file): 
    

    # next context
    c_end = df_source_file.end - end 
    c_start = df_source_file.start - end
    c1 = (c_end >= 0).values & (c_end<=10).values
    c2 = (c_start >= 0).values & (c_start <= 10).values
    c = c1 & c2
    
    next_context = df_source_file[c]
    
    c = (next_context.word == NS).values & (next_context.duration >= 0.4).values
    interruptions_next = next_context[c] 
    
    # prev context
    c_end = start - df_source_file.end
    c_start = start - df_source_file.start
    c1 = (c_start >=0).values & (c_start <= 10).values
    c2 = (c_end >= 0).values & (c_end<=10).values
    c = c1 & c2
    
    
    prev_context = df_source_file[c]
    c = (prev_context.word == NS).values & (prev_context.duration >= 0.4).values
    interruptions_prev = prev_context[c] 
    
    if interruptions_prev.empty:
        c = prev_context.word != NS
        prev_context = prev_context[c]
    else:
        c = prev_context.start >= interruptions_prev.iloc[-1].end
        prev_context = prev_context[c]
    if interruptions_next.empty:
        c = next_context.word != NS
        next_context = next_context[c]
    else:
        c = next_context.end <= interruptions_next.iloc[0].start
        next_context = next_context[c]
        
    return prev_context, next_context



""" librosa 
def get_audio_segments(row, audio, sr):
    prev_context=get_prev_context(row)
    next_context=get_next_context(row)
    #audio_file = get_file_path(row.source_file, '.wav')
    #audio, sr = librosa.load(audio_file, sr=None)
    # word
    
    word = row.word
    
    # prev
    if len(prev_context) > 0:
        onset = librosa.frames_to_samples(librosa.time_to_frames(prev_context.start.iloc[0], sr))
    else:
        onset = librosa.frames_to_samples(librosa.time_to_frames(row.start, sr))
    #offset = librosa.frames_to_samples(librosa.time_to_frames(prev_context.end.iloc[-1], sr))
    offset = librosa.frames_to_samples(librosa.time_to_frames(row.end, sr))
    audio_seg = audio[onset:offset]
    librosa.output.write_wav(os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0]+f'_{word}_prev.wav'), audio_seg, sr)
    
    # next
    # onset = librosa.frames_to_samples(librosa.time_to_frames(next_context.start.iloc[0], sr))
    onset = librosa.frames_to_samples(librosa.time_to_frames(row.start, sr))
    if len(next_context) > 0:
        offset = librosa.frames_to_samples(librosa.time_to_frames(next_context.end.iloc[-1], sr))
    else:
        offset = librosa.frames_to_samples(librosa.time_to_frames(row.end, sr))

    audio_seg = audio[onset:offset]
    librosa.output.write_wav(os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0]+f'_{word}_next.wav'), audio_seg, sr)
    
    prev_context= prev_context.append(row[['source_file', 'word', 'start', 'end', 'duration', 'label_type',
       'mp4_error', 'aac_error', 'aac2wav_error', 'eafgz_error', 'seg_error']])
    next_context = next_context.append(row[['source_file', 'word', 'start', 'end', 'duration', 'label_type',
       'mp4_error', 'aac_error', 'aac2wav_error', 'eafgz_error', 'seg_error']])
    
    return prev_context, os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0]+f'_{word}_prev.wav'), next_context, os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0]+f'_{word}_next.wav')
"""

def extract_interval_wave(audio, sr, start, end):
    """ 
    Returns wave data from start (seconds) to end (seconds) 
    
    Parameters
    ----------
    audio: wave data.
    sr: sampling rate of the audio.
    start: in seconds
    end: in seconds
    """
    start *= sr
    end *= sr
    return audio[np.int(start):np.int(end)]

def get_audio_segments(index,source_file, word, start, end, duration, df_source_file, audio, sr, save_files = True): 
   
    prev_context, next_context = get_context(source_file, word, start, end,df_source_file)

    # prev
    if len(prev_context) > 0:
        onset = prev_context.start.iloc[0]
    else:
        onset = start
     
    offset = end
    
    if save_files:
        audio_seg = extract_interval_wave(audio, sr,onset,offset)  
        wavfile.write(os.path.join(STRETCHES_PATH, source_file+'_%s_prev%s.wav' % (word,index)), sr, audio_seg)
        prev_file = os.path.join(STRETCHES_PATH, source_file+'_%s_prev%s.wav' % (word,index))
    else:
        prev_file = None
    
    # next
    if len(next_context) > 0:
        offset = next_context.end.iloc[-1]
    else:
        offset = end 

    
    if save_files:
        audio_seg = extract_interval_wave(audio, sr,onset,offset)
        wavfile.write(os.path.join(STRETCHES_PATH, source_file+'_%s_next%s.wav' % (word,index)), sr, audio_seg)
        next_file = os.path.join(STRETCHES_PATH, source_file+'_%s_next%s.wav' % (word,index))
    else:
        next_file = None
    
   
    
    prev_context.loc[len(prev_context)] = [source_file, word, start, end, duration, None, None, None, None, None, None, None]
    
    next_context.loc[len(next_context)] = [source_file, word, start, end, duration, None, None, None, None, None, None, None]
    
    return [prev_context, prev_file, next_context, next_file]


def detect_speaker_changes(speech_dection_model,peak_detection_model,file):
    test_file = {'audio': file}
    #print(file)
    try:
        speech_change_detection = speech_dection_model(test_file)
        partition = peak_detection_model.apply(speech_change_detection, dimension=1)
    except Exception as e:
        print(e)
        print("No partition found!")
        partition = None
    
    try:
        os.remove(file)
    except Exception as e:
        print(e)
    return partition


"""
def calculate_syl_counts(context):
    #missing_words = []
    #missing_words_estimate = []
    SylCnts = []
    for i,row in context.iterrows():
        if celex_dict[celex_dict.Word == row.word.lower()].PhonStrsDISC.empty:
            SyllCnt = syllables.estimate(row.word)
            #missing_words.append(row.word)
            #missing_words_estimate.append(SyllCnt)
        else:
            PhonStrsDisc = celex_dict[celex_dict.Word == row.word.lower()].PhonStrsDISC.iloc[0]
            SyllCnt = PhonStrsDisc.count("-")+1
        SylCnts.append(SyllCnt)

    return SylCnts
"""

def calculate_syl_counts(context): # words
    words_in_celex = context.word.str.lower().isin(celex_dict.Word.values).values
    #words_in_celex = np.isin([word.lower() for word in words], celex_dict.Word.values)
    words_not_in_celex = np.invert(words_in_celex) 
    
    """
    if np.sum(words_not_in_celex)>0:
        SylCnts_estimate = np.vectorize(syllables.estimate)(words[words_not_in_celex])
    else:
        SylCnts_estimate = []
    """
    SylCnts_estimate = context.word[words_not_in_celex].apply(syllables.estimate)
    
    SylCnts_celex = context[words_in_celex].merge(celex_dict[["PhonStrsDISC","Word"]],how="left",left_on = "word", right_on="Word").drop_duplicates()
    return np.sum(SylCnts_estimate) + np.sum(SylCnts_celex.PhonStrsDISC.str.count("-")+1)
    



def calculate_contextual_speaking_rate(row, use_heuristic = False):

    source_file, word, start, end = row[['source_file', 'word', 'start', 'end']]
    prev_context = row.prev_context
    next_context = row.next_context
    prev_partition = row.prev_partition
    next_partition = row.next_partition
    
    if use_heuristic:
        valid_prev_context = prev_context
        valid_next_context = next_context
        
    else:
        if prev_partition is None:
            speaker_onset = start
        else:
            speaker_onset = prev_partition[-1].end - prev_partition[-1].start
        if next_partition is None:
            speaker_offset = end
        else:
            speaker_offset = next_partition[0].end - next_partition[0].start

        c_start = start - prev_context.start
        valid_prev_c = (c_start >= 0).values & (c_start <= speaker_onset).values
        valid_prev_context = prev_context[valid_prev_c]
        
        c_end = next_context.end - end
        valid_next_c = (c_end >= 0).values & (c_end <= speaker_offset).values
        valid_next_context = next_context[valid_next_c]
    
    syl_counts_prev = calculate_syl_counts(valid_prev_context)
    syl_counts_next = calculate_syl_counts(valid_next_context)
    
    prev_stretch_duration = np.sort(valid_prev_context.end.values)[-1] - np.sort(valid_prev_context.start.values)[0]
    next_stretch_duration = np.sort(valid_next_context.end.values)[-1] - np.sort(valid_next_context.start.values)[0]
    
    
    speaking_rate_prev = syl_counts_prev/prev_stretch_duration
    speaking_rate_next = syl_counts_next/next_stretch_duration
    
    
    
    return pd.Series([speaking_rate_prev, speaking_rate_next, syl_counts_prev, syl_counts_next, prev_stretch_duration,next_stretch_duration ], 
                     ['speaking_rate_prev', 'speaking_rate_next', "syl_counts_prev", "syl_counts_next","prev_stretch_duration", "next_stretch_duration"])

    
def unfold_context(context_df_row):
    prev_context = context_df_row[0]
    prev_file = context_df_row[1]
    next_context = context_df_row[2]
    next_file = context_df_row[3]
    return prev_file, next_file, prev_context, next_context

unfold_context_vectorized = np.vectorize(unfold_context)    


def main(df_hom, 
         df_source,
         use_model = False,
         load_from_checkpoint = True, 
         checkpoint_file = 'done_source_files.txt', 
         checkpoint_speaking_rate_prev = "speaking_rates_prev.csv",
         checkpoint_speaking_rate_next = "speaking_rates_next.csv",
         checkpoint_syl_counts_prev = "syl_counts_prev.csv",
         checkpoint_syl_counts_next = "syl_counts_next.csv",
         checkpoint_prev_stretch_duration = "prev_stretch_duration.csv",
         checkpoint_next_stretch_duration = "next_stretch_duration.csv"
        ):
    
    if load_from_checkpoint:
        with open(checkpoint_file) as file:
        line = file.readlines()[0].split(",")
        last_i = int(line[0])
        source_file = line[1].strip("\n")
        
        remaining_source_files = df_hom.source_file.unique()[last_i+1:]
        
        speaking_rates_prev = pd.read_csv(checkpoint_speaking_rate_prev, index_col = "idx1")
        speaking_rates_next = pd.read_csv(checkpoint_speaking_rate_next, index_col = "idx1")
        syl_counts_prev = pd.read_csv(checkpoint_syl_counts_prev, index_col = "idx1")
        syl_counts_next = pd.read_csv(checkpoint_prev_stretch_duration, index_col = "idx1")
        prev_stretch_duration = pd.read_csv(checkpoint_prev_stretch_duration, index_col = "idx1")
        next_stretch_duration = pd.read_csv(checkpoint_next_stretch_duration, index_col = "idx1")
        
    else:
        last_i = -1
        remaining_source_files = df_hom.source_file.unique()
        
    df_hom.sort_values(by="source_file", inplace=True)
    n_unique_files = len(df_hom.source_file.unique())

    use_detection_model = False 

    get_audio_segments_vectorized = np.vectorize(get_audio_segments)
    calculate_contextual_speaking_rate_vectorized = np.vectorize(calculate_contextual_speaking_rate)

    if use_detection_model:
        detect_speaker_changes_vectorized = np.vectorize(lambda x: detect_speaker_changes(scd, peak, x))

    for f, source_file in enumerate(remaining_source_files):
        i = last_i+1+f
        start_time = timeit.default_timer()

        df_source_file = df_source[df_source.source_file == source_file]

        df_hom_file = df_hom[df_hom.source_file == source_file]

        idxs = df_hom_file.index.values
        source_files = df_hom_file.source_file.values
        words = df_hom_file.word.values
        starts =  df_hom_file.start.values
        ends =  df_hom_file.end.values
        durations = df_hom_file.duration.values


        if use_detection_model:
            audio_file = get_file_path(source_file, '.wav')
            sr, audio = wavfile.read(audio_file)    

            fn = np.vectorize(lambda idx,source_file,word,start,end,duration: get_audio_segments(idx,source_file, word, start, end, duration,df_source_file, audio, sr, save_files = True))

            context_df = fn(idxs,source_files, words, starts, ends, durations)

        else:
            fn = np.vectorize(lambda idx,source_file,word,start,end,duration: get_audio_segments(idx,source_file, word, start, end, duration,df_source_file, None, None, save_files = False))

 
            context_df = fn(idxs,source_files, words, starts, ends, durations)


        print("Done loading context")


       
        prev_files, next_files, prev_contexts, next_contexts = unfold_context_vectorized(context_df)



        if use_detection_model:
            print("Use detection Model!")
            detect_fn = lambda x: detect_speaker_changes(scd, peak, x)

          
            prev_partition = pd.Series(prev_files).apply(detect_fn)
            print("Done for prev..")
            next_partition = pd.Series(next_files).apply(detect_fn)
            s(scd, peak, x))
            print("Done for next..")


            speaking_rate_df = df_hom_file[['source_file', 'word', 'start', 'end']].copy()
            speaking_rate_df["prev_file"] = prev_files
            speaking_rate_df["next_file"] = next_files
            speaking_rate_df["prev_context"] = prev_contexts
            speaking_rate_df["next_context"] = next_contexts
            speaking_rate_df["prev_partition"] = list(prev_partition)
            speaking_rate_df["next_partition"] = list(next_partition)

            rates = speaking_rate_df.apply(lambda x: calculate_contextual_speaking_rate(x,use_heuristic = False), axis=1, raw=True)
            
        else:
            print("Use heuristic!")
            speaking_rate_df = df_hom_file[['source_file', 'word', 'start', 'end']].copy()
            speaking_rate_df["prev_file"] = prev_files
            speaking_rate_df["next_file"] = next_files
            speaking_rate_df["prev_context"] = prev_contexts
            speaking_rate_df["next_context"] = next_contexts
            speaking_rate_df["prev_partition"] = None
            speaking_rate_df["next_partition"] = None
            rates = speaking_rate_df.apply(lambda x: calculate_contextual_speaking_rate(x,use_heuristic = True), axis=1, raw=True)
            


        if i == 0:
            speaking_rates_prev = rates.speaking_rate_prev
            speaking_rates_next = rates.speaking_rate_next
            syl_counts_prev = rates.syl_counts_prev
            syl_counts_next = rates.syl_counts_next
            prev_stretch_duration = rates.prev_stretch_duration
            next_stretch_duration = rates.next_stretch_duration

        else:
            speaking_rates_prev = pd.concat([speaking_rates_prev, rates.speaking_rate_prev])
            speaking_rates_next = pd.concat([speaking_rates_next, rates.speaking_rate_next])
            syl_counts_prev = pd.concat([syl_counts_prev, rates.syl_counts_prev])
            syl_counts_next = pd.concat([syl_counts_next, rates.syl_counts_next])
            prev_stretch_duration = pd.concat([prev_stretch_duration, rates.prev_stretch_duration])
            next_stretch_duration = pd.concat([next_stretch_duration, rates.next_stretch_duration])

        end_time = timeit.default_timer()
        print("Finished calculating speaking rate for File %d/%d" % ((i+1), n_unique_files))
        print("Time: ", end_time - start_time)
        speaking_rates_prev.to_csv(checkpoint_speaking_rate_prev)
        speaking_rates_next.to_csv(checkpoint_speaking_rate_next)
        syl_counts_prev.to_csv(checkpoint_syl_counts_prev)
        syl_counts_next.to_csv(checkpoint_syl_counts_next)
        prev_stretch_duration.to_csv(checkpoint_prev_stretch_duration)
        next_stretch_duration.to_csv(checkpoint_next_stretch_duration)

        with open(checkpoint_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([i, source_file])
        file.close() 
        
        
if __name__ == '__main__':
    #pip install git+https://github.com/pyannote/pyannote-audio.git@develop
    #pip install pyAudioAnalysis
    
    df_source = pd.read_pickle(DF_SOURCE_PATH)
    df_hom = pd.read_csv(DF_HOMOGRAPHS_PATH, index_col = "Unnamed: 0")
    celex_dict = read_celex_file()
    scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
    peak = Peak(alpha=0.2, min_duration=0.20, log_scale=True)
     