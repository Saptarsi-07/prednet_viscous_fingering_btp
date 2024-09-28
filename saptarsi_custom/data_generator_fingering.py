import numpy as np 
import preprocess 
import math
import datetime # For logging into the terminal 
import warnings 
import os 

# Extract and preprocess frames from the video
frames = preprocess.preprocess("../fingering_bw_processed_128x128_30.mp4", (128, 128))

TRAIN_TEST_SPLIT_RATIO = 0.8 # How much data used for training? 
TRAIN_VALID_SPLIT_RATIO = 0.7 # How much training data used for validation
SIZE_TRAIN_VALID_DATA = math.ceil(len(frames) * TRAIN_TEST_SPLIT_RATIO)
SIZE_TRAIN_DATA = math.ceil(SIZE_TRAIN_VALID_DATA * TRAIN_VALID_SPLIT_RATIO)

print(f'Splitting data into {SIZE_TRAIN_DATA} training frames, {SIZE_TRAIN_VALID_DATA - SIZE_TRAIN_DATA} validation frames, and {len(frames) - SIZE_TRAIN_VALID_DATA} test frames')

train_data = frames[:SIZE_TRAIN_DATA]  # Split into training, validation, and test data 
valid_data = frames[SIZE_TRAIN_DATA:SIZE_TRAIN_VALID_DATA]
test_data = frames[SIZE_TRAIN_VALID_DATA:]

train_dict = dict()
valid_dict = dict()
test_dict = dict()

TOTAL_LENGTH = 20
INPUT_LENGTH = 10

LENGTH = SIZE_TRAIN_DATA//TOTAL_LENGTH
print(LENGTH)

INPUT_HEIGHT = 128
INPUT_WIDTH = 128
N_CHANNELS = 1 

# Initializing the train_dict to store clips and input data # 
train_dict['clips'] = np.zeros(shape=(2, LENGTH, 2), dtype=np.int32)
train_dict['input_raw_data'] = np.zeros(shape=(SIZE_TRAIN_DATA, N_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.int32)

for i in range(LENGTH):
    train_dict['clips'][0,i,0] = i * TOTAL_LENGTH
    train_dict['clips'][0,i,1] = INPUT_LENGTH
    train_dict['clips'][1,i,0] = i * TOTAL_LENGTH + INPUT_LENGTH
    train_dict['clips'][1,i,1] = INPUT_LENGTH
    for k in range(TOTAL_LENGTH):
        train_dict['input_raw_data'][i+k] = np.reshape(train_data[i+k], (N_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH))

# Store dimensions in the train_dict, not in train_data
train_dict['dims'] = [[N_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH]] # [[N_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH]]

# Print the stored data
print(train_dict['dims'])
print(train_dict['clips'].shape)
print(train_dict['input_raw_data'].shape)
print(train_data[0])

np.savez('../fingering_train_data.npz', **train_dict)

# Same for validation and test data...