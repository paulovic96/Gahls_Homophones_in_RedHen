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


def get_audio_segments(row):
    prev_context = get_prev_context(row)
    next_context = get_next_context(row)
    audio_file = get_file_path(row.source_file, '.wav')
    audio, sr = librosa.load(audio_file, sr=None)
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


def calulate_contextual_speaking_rate(row, prev_context, prev_partition, next_context, next_partition):
    source_file, word, start, end = row[['source_file', 'word', 'start', 'end']]
    speaker_onset = prev_partition[-1].end - prev_partition[-1].start
    speaker_offset = next_partition[0].end - next_partition[0].start

    # print(speaker_onset, start)

    valid_prev_context = prev_context.loc[
        (start - prev_context.start >= 0) &
        (start - prev_context.start <= speaker_onset)]

    # print(valid_prev_context)

    valid_next_context = next_context.loc[
        (next_context.end - end >= 0) &
        (next_context.end - end <= speaker_offset)]

    syl_counts_prev = [syllables.estimate(word) for word in
                       valid_prev_context.word]  # get_syl_counts(valid_prev_context).SylCnt
    syl_counts_next = [syllables.estimate(word) for word in
                       valid_next_context.word]  # get_syl_counts(valid_next_context).SylCnt
    print(np.sum(syl_counts_prev), np.sum(syl_counts_next))

    prev_stretch_duration = np.sort(valid_prev_context.end)[-1] - np.sort(valid_prev_context.start)[0]
    next_stretch_duration = np.sort(valid_next_context.end)[-1] - np.sort(valid_next_context.start)[0]

    print(prev_stretch_duration, next_stretch_duration)

    speaking_rate_prev = np.sum(syl_counts_prev) / prev_stretch_duration
    speaking_rate_next = np.sum(syl_counts_next) / next_stretch_duration

    return speaking_rate_prev, speaking_rate_next


if __name__ == '__main__':
    df_source = pd.read_pickle(DF_SOURCE_PATH)
    df_hom = pd.read_csv(DF_HOMEOHONES_PATH, index_col="Unnamed: 0")
    scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')  # , pipeline=True)
    peak = Peak(alpha=0.2, min_duration=0.20, log_scale=True)

    speaking_rates_prev = []
    speaking_rates_next = []

    counter = 0
    for idx, row in df_hom.iterrows():
        #row = df_hom.loc[0]
        if counter % 10000 == 0:
            print("Calculating speaking rate for row %d/%d" % (counter, len(df_hom)))
        prev_context, prev_file, next_context, next_file = get_audio_segments(row)

        prev_partition = detect_speaker_changes(scd, peak, prev_file)
        next_partition = detect_speaker_changes(scd, peak, next_file)

        speaking_rate_prev, speaking_rate_next = calulate_contextual_speaking_rate(row, prev_context, prev_partition, next_context, next_partition)
        speaking_rates_prev.append(speaking_rate_prev)
        speaking_rates_next.append(speaking_rate_next)

        os.remove(prev_file)
        os.remove(next_file)
        counter += 1

    df_hom["speaking_rate_prev"] = speaking_rates_prev
    df_hom["speaking_rate_next"] = speaking_rates_next

    df_hom.to_csv("2016_all_words_no_audio_preprocessed_speaking_rate.csv")

