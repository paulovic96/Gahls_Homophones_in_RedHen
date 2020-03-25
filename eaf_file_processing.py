# Author: Paul Schmidt-barbo and Elnaz Shafaei
import numpy as np
import os
import gzip
import poioapi.annotationgraph
import pandas as pd

FILE_EXT = '.eaf.gz'
LEN_FILE_EXT = len(FILE_EXT)

filepath = 'Data/eaf_files/2016-12-17_1330_US_KCET_Asia_Insight.eaf.gz'

#import shutil
#import gzip
#with open('Data/eaf_files/2016-12-17_1330_US_KCET_Asia_Insight.eaf', 'rb') as f_in:
#    with gzip.open('Data/eaf_files/2016-12-17_1330_US_KCET_Asia_Insight.eaf.gz', 'wb') as f_out:
#        shutil.copyfileobj(f_in, f_out)


def read_eaf(filepath):
    file = os.path.basename(filepath)[:-LEN_FILE_EXT]
    gesture_dict = {"time_region": [], "gesture": [], "file": []}  # store 1. Time Region, 2. Gesture, 3. File
    speech_annotation_dict = {"time_region": [], "annotation": [], "file": []}  # store 1. Time Region, 2. Annotation, 3. File

    # parsing ELAN's native .eaf file format with poio library
    ag = poioapi.annotationgraph.AnnotationGraph.from_elan(gzip.open(filepath))
    parser = poioapi.io.elan.Parser(gzip.open(filepath))

    # Get speech annotations and time_regions
    for annotation in ag.annotations_for_tier('ParentTier..Speech'):
        speech_annotation_dict["time_region"] += [parser.region_for_annotation(annotation)] # time_region
        speech_annotation_dict["annotation"] += [annotation.features['annotation_value']] # annotation
        speech_annotation_dict["file"] += [file] # file

    speech_annotation_eaf_data = pd.DataFrame.from_dict(speech_annotation_dict)

    # Get Machine annotated gesture and time_regions
    for tier in ag.tier_hierarchies:
        if "MachineAnnotation" in tier[0]: # ParenTier..Machineannotation
            for child in tier[1:]: # iterate over child tiers (ParentTier..Machineannotation first element in list)
                if len(child) > 1: # further child tiers
                    gesture = child[0].split("..")[1] # get tier description
                    for child_child in child[1:]: # iterate over grandchildren tiers
                        sub_gesture = child_child[0].split("..")[1] # get tier description
                        key = gesture + "/" + sub_gesture # add to root tier description
                        for annotation in ag.annotations_for_tier(child_child[0]):
                            gesture_dict["time_region"] += [parser.region_for_annotation(annotation)] # time_region
                            gesture_dict["gesture"] += [key] # gesture
                            gesture_dict["file"] += [file] # file
                else:
                    key = child[0].split("..")[1] # get tier description
                    for annotation in ag.annotations_for_tier(child[0]):
                            gesture_dict["time_region"] += [parser.region_for_annotation(annotation)] # time_region
                            gesture_dict["gesture"] += [key] # gesture
                            gesture_dict["file"] += [file] # file

    gesture_eaf_data = pd.DataFrame.from_dict(gesture_dict)

    # Add Start and End point of Time region
    start = []
    end = []
    for i in gesture_eaf_data["time_region"]:
        region = i #eval(i)
        start.append(region[0])
        end.append(region[1])

    gesture_eaf_data["start"] = start
    gesture_eaf_data["end"] = end

    start = []
    end = []

    for i in speech_annotation_eaf_data["time_region"]:
        region = i #eval(i)
        start.append(region[0])
        end.append(region[1])

    speech_annotation_eaf_data["start"] = start
    speech_annotation_eaf_data["end"] = end

    gesture_eaf_data = gesture_eaf_data.drop(columns=['time_region'])
    speech_annotation_eaf_data = speech_annotation_eaf_data.drop(columns=['time_region'])

    return (speech_annotation_eaf_data, gesture_eaf_data)

def map_gestures_to_annotation(speech_annotation_eaf_data, gesture_eaf_data, remove_pauses=True):
    # create time point tuple for line sweep algorithm for each file
    # Tuple Structure:
    # - Time point
    # - Annotation Value
    # - Info whether it is a start or end point
    # - Info whether it is a speech annotation or gesture

    file_time_points = {}

    for file in np.unique(gesture_eaf_data["file"]):
        speech_annotation_file_i = speech_annotation_eaf_data[speech_annotation_eaf_data["file"] == file].copy()
        gesture_annotation_file_i = gesture_eaf_data[gesture_eaf_data["file"] == file].copy()

        # create Time point tuple with info about 1. time point 2. annotation value 3. Start or End point 4. Speech annotation or gesture
        start_speech_time_points_i = list(zip(list(speech_annotation_file_i["start"]), list(speech_annotation_file_i["annotation"]),np.repeat("start", len(speech_annotation_file_i)), np.repeat("annotation", len(speech_annotation_file_i))))
        end_speech_time_points_i = list(zip(list(speech_annotation_file_i["end"]), list(speech_annotation_file_i["annotation"]),np.repeat("end", len(speech_annotation_file_i)), np.repeat("annotation", len(speech_annotation_file_i))))

        # alternating insertion to ensure start-end order
        annotation_time_points_i = [None]*(len(start_speech_time_points_i)+len(end_speech_time_points_i))
        annotation_time_points_i[::2] = start_speech_time_points_i
        annotation_time_points_i[1::2] = end_speech_time_points_i

        start_gesture_time_points_i = list(zip(list(gesture_annotation_file_i["start"]), list(gesture_annotation_file_i["gesture"]),np.repeat("start", len(gesture_annotation_file_i)), np.repeat("gesture", len(gesture_annotation_file_i))))
        end_gesture_time_points_i = list(zip(list(gesture_annotation_file_i["end"]), list(gesture_annotation_file_i["gesture"]),np.repeat("end", len(gesture_annotation_file_i)), np.repeat("gesture", len(gesture_annotation_file_i))))

        # add all time points to one list and sort by time
        time_points = annotation_time_points_i + start_gesture_time_points_i + end_gesture_time_points_i
        time_points.sort(key=lambda x: x[0])

        # add sorted time points to file dic
        file_time_points[file] = time_points

    # Line Sweep Algorithm to keep track of active annotations over time
    # for each file hold lists of active annotations and gestures
    # for each time_point add current file, time_point, annotation and list of gestures to dataset

    annotation_gesture_dict = {"file":[], "time_point":[], "annotation":[], "gesture": []}

    for current_file in file_time_points.keys(): # iterate over time points
        current_file_time_points = file_time_points[current_file] # get time points
        active_gestures = [] # currently (at given time point) active gestures
        active_annotations = [] # currently (at given time point) active annotations
        for time_point in current_file_time_points: # iterate over time points
            if time_point[2] == 'start':
                if time_point[3] == 'annotation':
                    active_annotations.append(time_point[1]) # new annotations
                else:
                    active_gestures.append(time_point[1]) # new gesture
            else:
                if time_point[3] == 'annotation':
                    active_annotations.remove(time_point[1]) # annotation finished
                else:
                    active_gestures.remove(time_point[1]) # gesture finished

            # info at each point in time
            annotation_gesture_dict["file"].append(current_file)
            annotation_gesture_dict["time_point"].append(time_point)
            annotation_gesture_dict["annotation"].append(list(active_annotations))
            annotation_gesture_dict["gesture"].append(list(active_gestures))

    annotation_gesture_eaf_data = pd.DataFrame.from_dict(annotation_gesture_dict)

    annotation_gesture_eaf_data["annotation"] = annotation_gesture_eaf_data["annotation"].apply(lambda y: np.nan if len(y)==0 else y)
    annotation_gesture_eaf_data["gesture"] = annotation_gesture_eaf_data["gesture"].apply(lambda y: np.nan if len(y)==0 else y)

    if remove_pauses:
        # Only keep rows with speech annotation
        annotation_gesture_eaf_data.dropna(subset=['annotation', 'gesture'], how='all')
        annotation_gesture_eaf_data = annotation_gesture_eaf_data[pd.notnull(annotation_gesture_eaf_data["annotation"])]

    # If more than one annotation value at time point (some inprecision in time points start of another annotation before first one ended) keep newly started one

    annotation_gesture_eaf_data["annotation"] = annotation_gesture_eaf_data["annotation"].apply(lambda y: (y[0] if len(y)==1 else y[1]) if isinstance(y, list) else y)

    # only keep start points of annotation (only interesting points)
    indices = []
    valid_start_points = []
    for i, point in enumerate(annotation_gesture_eaf_data["time_point"]):
        if point[2] == "start":
            if point[3] == "annotation":
                valid_start_points.append(point[0])
                indices.append(annotation_gesture_eaf_data.index[i])
            else :
                if pd.isnull(annotation_gesture_eaf_data["annotation"].iloc[i]): #valid pause if included than point is important otherwise we would add multiple time points for same annotation because gesture still active
                    valid_start_points.append(point[0])
                    indices.append(annotation_gesture_eaf_data.index[i])





    annotation_gesture_eaf_data = annotation_gesture_eaf_data[annotation_gesture_eaf_data.index.get_level_values(0).isin(indices)]
    annotation_gesture_eaf_data["start"] = valid_start_points

    merged_annotation_gesture_eaf_data = speech_annotation_eaf_data.merge(annotation_gesture_eaf_data[["file", "annotation", "gesture", "start", "time_point"]], on=["file", "start", "annotation"], how = "outer")

    return merged_annotation_gesture_eaf_data



difference = merged_annotation_gesture_eaf_data.merge(speech_annotation_eaf_data, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']

