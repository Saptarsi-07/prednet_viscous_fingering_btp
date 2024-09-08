author = 'recklurker'

"""
This file contains logic for greyscaling, rescaling to custom resolution (for example 128x128) using OpenCV (cv2)
"""

import cv2
import os 
import numpy as np 
from terminal_colors import terminal_colors

def extract_frames(path_to_video):
    '''
        extract_frames(path_to_video)
        Extracts the frames from the video available at path. 
    '''
    vidcap = cv2.VideoCapture(path_to_video)
    count = 0 # Used to count how many frames has been extracted 
    success = True # Used to check if a frame has been extracted successfully 
    frames = []
    while success: 
        success, frame = vidcap.read();
        if not success and count == 0: 
            print(f'{terminal_colors.WARNING}Reading Video File failed! Check File Path.{terminal_colors.ENDC}')
            exit()
        elif not success: 
            continue 
        frames.append(frame) # append extracted frame 
        count += 1
    return np.array(frames)

def rescale_frames(frames, resolution):
    for frame in frames: 
        frame = cv2.resize(frame, tuple(resolution))

def grayscale_frames(frames):
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def preprocess(path_to_video, resolution): 
    '''
        preprocess(path_to_video)
        Preprocess the video available at path_to_video and writes the output at output_folder
    '''

    if len(resolution) != 2:
        print(f'{terminal_colors.WARNING}Expected 2 values, got {len(resolution)}{terminal_colors.ENDC}')

    frames = extract_frames(path_to_video)
    rescale_frames(frames, resolution)
    grayscale_frames(frames)

    # if not output_folder[-1] == "\\":
    #     output_folder = output_folder + '\\'
    # if os.path.isdir(output_folder):
    #     os.rmdir(output_folder)
    # os.mkdir(output_folder)
    # count = 0
    # for frame in frames:
    #     count += 1
    #     path = output_folder + f"{count:04}" + ".jpg"
    #     cv2.imwrite(path, frame)
    return frames 

