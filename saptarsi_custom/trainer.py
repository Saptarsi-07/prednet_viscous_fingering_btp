import numpy as np 
import preprocess 
import math
import model as predRNNv2


frames = preprocess.preprocess("../fingering_bw_processed_128x128_30.mp4", (128, 128))

TRAIN_TEST_SPLIT_RATIO = 0.8 
SIZE_TRAIN_DATA = math.ceil(len(frames) * TRAIN_TEST_SPLIT_RATIO)

train_data = frames[:SIZE_TRAIN_DATA+1]
test_data = frames[SIZE_TRAIN_DATA+1:]

# Model Configuration
config = {
    'is_training': 0,
    'device': 'cuda',
    'model_name': 'predRNNv2',
    'visual': 0,
    'reverse_input': 1,
    'img_width': 128,
    'img_channel': 1,
    'input_length': 10,
    'total_length': 20,
    'num_hidden': [128, 128, 128, 128],
    'filter_size': 5,
    'stride': 1,
    'patch_size': 4,
    'layer_norm': 0,
    'decouple_beta': 0.01,
    'reverse_scheduled_sampling': 1,
    'r_sampling_step_1': 5000,
    'r_sampling_step_2': 50000,
    'r_exp_alpha': 2000,
    'lr': 0.0001,
    'batch_size': 4,
    'max_iterations': 80000,
    'display_interval': 100,
    'test_interval': 5000,
    'snapshot_interval': 5000
}

model = predRNNv2.Model(config)