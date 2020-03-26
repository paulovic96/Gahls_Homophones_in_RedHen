import numpy as np
import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


FILE_EXT = '.mp4'
LEN_FILE_EXT = len(FILE_EXT)

filepath = 'Data/video_files/2016-12-17_1330_US_KCET_Asia_Insight.mp4'


def get_word_video_snippet_size(data, filepath):
    file = os.path.basename(filepath)[:-LEN_FILE_EXT]
    video_snippet_dict = {"source_file": [], "start": [], "end": [], "video_snippet_size": []}

    current_video_data = data[data["source_file"] == file]

    for index, row in current_video_data.iterrows():
        video_snippet_dict["source_file"].append(file)
        video_snippet_dict["start"].append(row["start"])
        video_snippet_dict["end"].append(row["end"])

        start_sec = row["start"] / 1000
        end_sec = row["end"] / 1000

        # print(start_sec, end_sec)
        print(index, row["annotation"])
        ffmpeg_extract_subclip(filepath, start_sec, end_sec, targetname="snippet.mp4")

        # print(os.path.getsize("test.mp4"))
        video_snippet_dict["video_snippet_size"].append(os.path.getsize("snippet.mp4"))
        os.remove("snippet.mp4")

    return video_snippet_dict

test = merged_annotation_gesture_eaf_data.sort_values(by = "start",ignore_index=True)
video_snippet_dict = get_word_video_snippet_size(test, filepath)
