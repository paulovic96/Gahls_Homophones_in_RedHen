# Author: Paul Schmidt-barbo and Elnaz Shafaeis
import numpy as np
import os
import gzip
import poioapi.annotationgraph
import pandas as pd


FILE_EXT = '.eaf.gz'
LEN_FILE_EXT = len(FILE_EXT)

#filepath = '/mnt/Restricted/Corpora/RedHen/original/2016/2016-01/2016-01-01/2016-01-01_0100_US_KNBC_Channel_4_News.eaf.gz'


def read_eaf(filepath):
    """
    :param filepath (str): the eaf file
    :return speech_annotation_eaf_data (pd.DataFrame): Dataframe containing the annotated words with time points
    :return gesture_eaf_data (pd.DataFrame): Dataframe containing the annotated gestures with time points
    """
    file = os.path.basename(filepath)[:-LEN_FILE_EXT] # file name without ending
    gesture_dict = {"time_region": [], "gesture": [], "source_file": []}  # store 1. Time Region, 2. Gesture, 3. File
    speech_annotation_dict = {"time_region": [], "annotation": [], "source_file": []}  # store 1. Time Region, 2. Annotation, 3. File

    # parsing ELAN's native .eaf file format with poio library
    ag = poioapi.annotationgraph.AnnotationGraph.from_elan(gzip.open(filepath))
    parser = poioapi.io.elan.Parser(gzip.open(filepath))

    # Get speech annotations and time_regions
    for annotation in ag.annotations_for_tier('ParentTier..Speech'):
        speech_annotation_dict["time_region"] += [parser.region_for_annotation(annotation)] # time_region
        speech_annotation_dict["annotation"] += [annotation.features['annotation_value']] # annotation
        speech_annotation_dict["source_file"] += [file] # file

    speech_annotation_eaf_data = pd.DataFrame.from_dict(speech_annotation_dict) # Dataframe contianin the words annotated in the eaf file

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
                            gesture_dict["source_file"] += [file] # file
                else:
                    key = child[0].split("..")[1] # get tier description
                    for annotation in ag.annotations_for_tier(child[0]):
                            gesture_dict["time_region"] += [parser.region_for_annotation(annotation)] # time_region
                            gesture_dict["gesture"] += [key] # gesture
                            gesture_dict["source_file"] += [file] # file

    gesture_eaf_data = pd.DataFrame.from_dict(gesture_dict) # Dataframe containing the gestures annotated in the eaf file

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

    #gesture_eaf_data = gesture_eaf_data.drop(columns=['time_region'])
    #speech_annotation_eaf_data = speech_annotation_eaf_data.drop(columns=['time_region'])

    return (speech_annotation_eaf_data, gesture_eaf_data)

def map_gestures_to_annotation(speech_annotation_eaf_data, gesture_eaf_data, remove_pauses=True):
    """
    :param speech_annotation_eaf_data (pd.DataFrame): Dataframe containing the annotated words with time points
    :param gesture_eaf_data (pd.DataFrame): Dataframe containing the annotated gestures with time points
    :param remove_pauses (boolean): whether we want to remove time points where nothing is said
    :return merged_annotation_gesture_eaf_data (pd.DataFrame): Dataframe containing the annotated speech plus information about used gestures during articulation
    """

    # create time point tuple for line sweep algorithm for each file
    # Tuple Structure:
    # - Time point
    # - Annotation Value
    # - Info whether it is a start or end point
    # - Info whether it is a speech annotation or gesture

    file_time_points = {}

    for file in np.unique(gesture_eaf_data["source_file"]):
        speech_annotation_file_i = speech_annotation_eaf_data[speech_annotation_eaf_data["source_file"] == file].copy()
        gesture_annotation_file_i = gesture_eaf_data[gesture_eaf_data["source_file"] == file].copy()

        # create Time point tuple with info about 1. time point 2. annotation value 3. Start or End point 4. Speech annotation or gesture
        start_speech_time_points_i = list(zip(list(speech_annotation_file_i["start"]), list(speech_annotation_file_i["annotation"]),np.repeat("start", len(speech_annotation_file_i)), np.repeat("annotation", len(speech_annotation_file_i))))
        end_speech_time_points_i = list(zip(list(speech_annotation_file_i["end"]), list(speech_annotation_file_i["annotation"]),np.repeat("end", len(speech_annotation_file_i)), np.repeat("annotation", len(speech_annotation_file_i))))

        # alternating insertion to ensure start-end order
        annotation_time_points_i = [None]*(len(start_speech_time_points_i)+len(end_speech_time_points_i))
        annotation_time_points_i[::2] = start_speech_time_points_i
        annotation_time_points_i[1::2] = end_speech_time_points_i

        annotation_time_regions_i = list(np.repeat(speech_annotation_file_i["time_region"],2))

        start_gesture_time_points_i = list(zip(list(gesture_annotation_file_i["start"]), list(gesture_annotation_file_i["gesture"]),np.repeat("start", len(gesture_annotation_file_i)), np.repeat("gesture", len(gesture_annotation_file_i))))
        end_gesture_time_points_i = list(zip(list(gesture_annotation_file_i["end"]), list(gesture_annotation_file_i["gesture"]),np.repeat("end", len(gesture_annotation_file_i)), np.repeat("gesture", len(gesture_annotation_file_i))))

        gesture_time_regions_i = list(np.repeat(gesture_annotation_file_i["time_region"],2))

        # add all time points to one list and sort by time
        time_points = annotation_time_points_i + start_gesture_time_points_i + end_gesture_time_points_i
        time_points.sort(key=lambda x: x[0])

        time_regions = annotation_time_regions_i + gesture_time_regions_i
        time_regions.sort(key=lambda x: x[0])


        # add sorted time points to file dic
        file_time_points[file] = (time_points,time_regions)

    # Line Sweep Algorithm to keep track of active annotations over time
    # for each file hold lists of active annotations and gestures
    # for each time_point add current file, time_point, annotation and list of gestures to dataset

    annotation_gesture_dict = {"source_file":[], "time_point":[], "annotation":[], "gesture": [], "time_region": []}

    for current_file in file_time_points.keys(): # iterate over time points
        current_file_time_points = file_time_points[current_file][0] # get time points
        current_file_time_regions = file_time_points[current_file][1] # get time regions (for later checking and adding of end points if pauses not removed)
        active_gestures = [] # currently (at given time point) active gestures
        active_annotations = [] # currently (at given time point) active annotations
        for i,time_point in enumerate(current_file_time_points): # iterate over time points
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
            annotation_gesture_dict["source_file"].append(current_file)
            annotation_gesture_dict["time_region"].append(current_file_time_regions[i])
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

    merged_annotation_gesture_eaf_data = speech_annotation_eaf_data.merge(annotation_gesture_eaf_data[["source_file", "annotation", "gesture", "start", "time_point", "time_region"]], on=["source_file", "start", "annotation"], how = "outer")

    merged_annotation_gesture_eaf_data.sort_values(by="start", inplace=True, ignore_index=True)
    merged_annotation_gesture_eaf_data.loc[pd.notnull(merged_annotation_gesture_eaf_data["time_region_x"]),"time_region_y"] = np.nan
    merged_annotation_gesture_eaf_data.rename(columns={"time_region_y": "time_region_gesture"}, inplace = True)
    merged_annotation_gesture_eaf_data.drop(columns=['time_region_x'], inplace = True)



    # fill in missing end points for pauses = no annotation but gesture present
    if not remove_pauses:
        valid_end_points = []
        active_gesture = ""
        active_end_region = 0

        for index, row in merged_annotation_gesture_eaf_data.iterrows():
            if pd.isnull(row["end"]):
                if len(str(active_gesture)) == 0:
                    active_gesture = row["gesture"]
                    active_end_region = row["time_region_gesture"][1]
                else : # new gesture present
                    if row["start"] <= active_end_region:
                        valid_end_points.append(row["start"])
                    else:
                        valid_end_points.append(active_end_region)
                    active_gesture = row["gesture"]
                    active_end_region = row["time_region_gesture"][1]
            else: # annoation
                if len(str(active_gesture)) == 0:
                    valid_end_points.append(row["end"])
                else:
                    if row["start"] <= active_end_region:
                        valid_end_points.append(row["start"])
                        valid_end_points.append(row["end"])

                    else:
                        valid_end_points.append(active_end_region)
                        valid_end_points.append(row["end"])
                    active_gesture = ""
                    active_end_region = 0

        if len(active_gesture)>0:
            valid_end_points.append(active_end_region)

        merged_annotation_gesture_eaf_data["end"] = valid_end_points


    return merged_annotation_gesture_eaf_data


def binary_encode_gestures(data, gesture_column = "gesture"):
    """
    :param data (pd.Dataframe): Dataframe containig speech annotation with already matched gestures
    :param gesture_column (str): Name of the column containing the gesture information
    :return (pd.DataFrame): data joined with columns for binary encoded gestures
    """
    unique_gestures = set()
    for gesture_list in data[gesture_column]:
        if isinstance(gesture_list, list):
            for gesture in gesture_list:
                unique_gestures.add(gesture)
        else:
            unique_gestures.add("none")

    binary_gestures = {}
    for gesture in unique_gestures:
        binary_gestures.update({gesture: []})

    for gesture_list in data[gesture_column]:
        if isinstance(gesture_list, list):
            for key in binary_gestures.keys():
                if key in gesture_list:
                    binary_gestures[key] += [1]
                else:
                    binary_gestures[key] += [0]

        else:
            for key in binary_gestures.keys():
                if key != "none":
                    binary_gestures[key] += [0]
                else:
                    binary_gestures["none"] += [1]

    binary_gesture_data = pd.DataFrame.from_dict(binary_gestures)
    binary_gesture_data["is_gesture"] = binary_gesture_data["none"] == 0

    return data.join(binary_gesture_data)


