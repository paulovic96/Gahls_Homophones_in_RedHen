import os
import pandas as pd
from celex_files import read_celex_file, get_syl_counts

BASE = '/mnt/Restricted/Corpora/RedHen'
DATA_FOLDER = os.path.join(BASE, 'original')
DF_SOURCE_PATH = os.path.join(BASE, '2016_all_words_no_audio.pickle')
DF_HOMEOHONES_PATH = os.path.join(BASE, 'homophone_analysis_scripts/2016_all_words_no_audio_preprocessed.csv')

STRETCHES_PATH = '/mnt/shared/people/elnaz/homophones/10sec_stretch'

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

if __name__ == '__main__':
    df_source = pd.read_pickle(DF_SOURCE_PATH)
    df_hom = pd.read_csv(DF_HOMEOHONES_PATH)
    row = df_hom.loc[0]
    prev_context=get_prev_context(row)
    next_context=get_next_context(row)
    audio_file = get_file_path(row.source_file, '.wav')
    audio, sr = librosa.load(audio_file, sr=None)
    # word
    word = row.word
    # prev
    onset = librosa.frames_to_samples(librosa.time_to_frames(prev_context.start.iloc[0], sr))
    # offset = librosa.frames_to_samples(librosa.time_to_frames(prev_context.end.iloc[-1], sr))
    offset = librosa.frames_to_samples(librosa.time_to_frames(row.end, sr))
    audio_seg = audio[onset:offset]
    librosa.output.write_wav(os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0]+f'_{word}_prev.wav'), audio_seg, sr)
    # next
    # onset = librosa.frames_to_samples(librosa.time_to_frames(next_context.start.iloc[0], sr))
    onset = librosa.frames_to_samples(librosa.time_to_frames(row.start, sr))
    offset = librosa.frames_to_samples(librosa.time_to_frames(next_context.end.iloc[-1], sr))
    audio_seg = audio[onset:offset]
    librosa.output.write_wav(os.path.join(STRETCHES_PATH, df_hom.source_file.iloc[0]+f'_{word}_next.wav'), audio_seg, sr)
