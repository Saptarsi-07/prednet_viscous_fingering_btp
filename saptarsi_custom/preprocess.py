author = 'recklurker'

"""
This file contains logic for greyscaling, rescaling to custom resolution (for example 128x128) using OpenCV (cv2)
"""

import cv2
import os 
import numpy as np 

def preprocess(path_to_video, resolution):
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
    framesTemp = np.array(frames)
    framesRes = []
    for frame in framesTemp: 
        frame = cv2.resize(frame, tuple(resolution))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        framesRes.append(frame)
    return np.array(framesRes)
