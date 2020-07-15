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


def get_audio_segments(row,audio,sr):
    prev_context = get_prev_context(row)
    next_context = get_next_context(row)
    #audio_file = get_file_path(row.source_file, '.wav')
    #audio, sr = librosa.load(audio_file, sr=None)
    # word

    word = row.word

    # prev
    if len(prev_context) > 0:
        onset = librosa.frames_to_samples(librosa.time_to_frames(prev_context.start.iloc[0], sr))
    else:
        onset = librosa.frames_to_samples(librosa.time_to_frames(row.start, sr))
    # offset = librosa.frames_to_samples(librosa.time_to_frames(prev_context.end.iloc[-1], sr))
    offset = librosa.frames_to_samples(librosa.time_to_frames(row.end, sr))
    audio_seg = audio[onset:offset]
    librosa.output.write_wav(os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0] + f'_{word}_prev.wav'), audio_seg,
                             sr)

    # next
    # onset = librosa.frames_to_samples(librosa.time_to_frames(next_context.start.iloc[0], sr))
    onset = librosa.frames_to_samples(librosa.time_to_frames(row.start, sr))
    if len(next_context) > 0:
        offset = librosa.frames_to_samples(librosa.time_to_frames(next_context.end.iloc[-1], sr))
    else:
        offset = librosa.frames_to_samples(librosa.time_to_frames(row.end, sr))

    audio_seg = audio[onset:offset]
    librosa.output.write_wav(os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0] + f'_{word}_next.wav'), audio_seg,
                             sr)

    prev_context = prev_context.append(row[['source_file', 'word', 'start', 'end', 'duration', 'label_type',
                                            'mp4_error', 'aac_error', 'aac2wav_error', 'eafgz_error', 'seg_error']])
    next_context = next_context.append(row[['source_file', 'word', 'start', 'end', 'duration', 'label_type',
                                            'mp4_error', 'aac_error', 'aac2wav_error', 'eafgz_error', 'seg_error']])

    return prev_context, os.path.join(STRETCHES_PATH,
                                      df_hom.source_file.iloc[0] + f'_{word}_prev.wav'), next_context, os.path.join(
        STRETCHES_PATH, df_hom.source_file.iloc[0] + f'_{word}_next.wav')


def detect_speaker_changes(speech_dection_model, peak_detection_model, file):
    test_file = {'audio': file}
    speech_change_detection = speech_dection_model(test_file)

    partition = peak_detection_model.apply(speech_change_detection, dimension=1)
    return partition


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
    

def calculate_contextual_speaking_rate(row, prev_context,prev_partition,next_context,next_partition):
    source_file, word, start, end = row[['source_file', 'word', 'start', 'end']]
    speaker_onset = prev_partition[-1].end - prev_partition[-1].start
    speaker_offset = next_partition[0].end - next_partition[0].start
    
    
    valid_prev_context = prev_context.loc[
                                 (start - prev_context.start >= 0) & 
                                 (start - prev_context.start <= speaker_onset) ]
    
    
    valid_next_context = next_context.loc[
                                 (next_context.end - end >= 0) & 
                                 (next_context.end - end <= speaker_offset)]
    
    #syl_counts_prev = [syllables.estimate(word) for word in valid_prev_context.word]#get_syl_counts(valid_prev_context).SylCnt
    #syl_counts_next = [syllables.estimate(word) for word in valid_next_context.word]#get_syl_counts(valid_next_context).SylCnt
    syl_counts_prev = calculate_syl_counts(valid_prev_context)
    syl_counts_next = calculate_syl_counts(prev_next_context)
    
    prev_stretch_duration = np.sort(valid_prev_context.end)[-1] - np.sort(valid_prev_context.start)[0]
    next_stretch_duration = np.sort(valid_next_context.end)[-1] - np.sort(valid_next_context.start)[0]
    
    
    speaking_rate_prev = np.sum(syl_counts_prev)/prev_stretch_duration
    speaking_rate_next = np.sum(syl_counts_next)/next_stretch_duration
    
    return speaking_rate_prev, speaking_rate_next

if __name__ == '__main__':
    df_source = pd.read_pickle(DF_SOURCE_PATH)
    df_hom = pd.read_csv(DF_HOMEOHONES_PATH, index_col="Unnamed: 0")
    scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')  # , pipeline=True)
    peak = Peak(alpha=0.2, min_duration=0.20, log_scale=True)
    
    df_hom.sort_values(by="source_file", inplace=True)
    
    speaking_rates_prev = []
    speaking_rates_next = []
    
    
    counter = 0
    file_counter = 1
    n_unique_files = len(df_hom.source_file.unique())
    n_rows = len(df_hom)
    
    for idx, row in df_hom.iterrows():
        if counter == 0: 
            print("File %d/%d" % (file_counter,n_unique_files))
            current_file = row.source_file
            audio_file = get_file_path(current_file, '.wav')
            audio, sr = librosa.load(audio_file, sr=None)

        else:
            if row.source_file != current_file: 
                file_counter += 1
                print("File %d/%d" % (file_counter,n_unique_files))
                current_file = row.source_file 
                audio_file = get_file_path(current_file, '.wav')
                audio, sr = librosa.load(audio_file, sr=None)


        prev_context, prev_file, next_context, next_file = get_audio_segments(row, audio, sr)

        prev_partition = detect_speaker_changes(scd, peak, prev_file)
        next_partition = detect_speaker_changes(scd, peak, next_file)

        speaking_rate_prev, speaking_rate_next = calulate_contextual_speaking_rate(row, prev_context, prev_partition, next_context, next_partition)
        os.remove(prev_file)
        os.remove(next_file)

        if counter % 1000 == 0:
            print("Finished calculating speaking rate for row %d/%d" % (counter,n_rows))

        counter += 1
    
    
    df_hom["speaking_rate_prev"] = speaking_rates_prev
    df_hom["speaking_rate_next"] = speaking_rates_next

    df_hom.to_csv("2016_all_words_no_audio_preprocessed_speaking_rate.csv")

