import os
import pandas as pd
import subprocess as sp
import proglog

from moviepy.config import get_setting
#from moviepy.tools import subprocess_call
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000 * t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [get_setting("FFMPEG_BINARY"), "-hide_banner", "-loglevel", "warning", "-y",
           "-ss", "%0.2f" % t1,
           "-i", filename,
           "-t", "%0.2f" % (t2 - t1),
           "-map", "0", "-vcodec", "copy", "-acodec", "copy", targetname]

    subprocess_call(cmd)


def subprocess_call(cmd, logger='bar', errorprint=True):
    """ Executes the given subprocess command.

    Set logger to None or a custom Proglog logger to avoid printings.
    """
    logger = proglog.default_bar_logger(logger)
    #logger(message='Moviepy - Running:\n>>> "+ " ".join(cmd)')

    popen_params = {"stdout": sp.DEVNULL,
                    "stderr": sp.PIPE,
                    "stdin": sp.DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)

    out, err = proc.communicate()  # proc.wait()
    proc.stderr.close()

    if proc.returncode:
        if errorprint:
            logger(message='Moviepy - Command returned an error')
        raise IOError(err.decode('utf8'))
    #else:
        #logger(message='Moviepy - Command successful')

    del proc


FILE_EXT = '.mp4'
LEN_FILE_EXT = len(FILE_EXT)

#filepath = 'Data/video_files/2016-12-17_1330_US_KCET_Asia_Insight.mp4', 'Data/video_files/2016-10-25_2300_US_KABC_Eyewitness_News_4PM.mp4'


def get_word_video_snippet_size(data, filepath):
    file = os.path.basename(filepath)[:-LEN_FILE_EXT]
    video_snippet_dict = {"source_file": [],"word":[], "start": [], "end": [], "video_snippet_size": []}

    current_video_data = data[data["source_file"] == file]

    for index, row in current_video_data.iterrows():
        video_snippet_dict["source_file"].append(file)
        video_snippet_dict["word"].append(row["word"])
        video_snippet_dict["start"].append(row["start"])
        video_snippet_dict["end"].append(row["end"])

        start_sec = row["start"] / 1000
        end_sec = row["end"] / 1000

        ffmpeg_extract_subclip(filepath, start_sec, end_sec, targetname="snippet.mp4")

        # print(os.path.getsize("test.mp4"))
        video_snippet_dict["video_snippet_size"].append(os.path.getsize("snippet.mp4"))
        os.remove("snippet.mp4")

    return pd.DataFrame.from_dict(video_snippet_dict)

