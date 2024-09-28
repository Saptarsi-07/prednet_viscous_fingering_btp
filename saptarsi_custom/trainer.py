import numpy as np 
import preprocess 
import math
import model as predRNNv2
import datetime # For logging into the terminal 
import warnings 
import os 

# Extract and preprocess frames from the video # 
frames = preprocess.preprocess("../fingering_bw_processed_128x128_30.mp4", (128, 128))

TRAIN_TEST_SPLIT_RATIO = 0.8 # How much data used for training? 
SIZE_TRAIN_DATA = math.ceil(len(frames) * TRAIN_TEST_SPLIT_RATIO)

print(f'Splitting data into {SIZE_TRAIN_DATA} training frames and {len(frames) - SIZE_TRAIN_DATA} test frames')
train_data = frames[:SIZE_TRAIN_DATA+1] # Split into training and testing data # 
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
    'batch_size': 5,
    'max_iterations': 80000,
    'display_interval': 100,
    'test_interval': 5000,
    'snapshot_interval': 5000
}

ITER_PER_BATCH = SIZE_TRAIN_DATA // config['batch_size']

print('Initializing Model')
model = predRNNv2.Model(config)

# Used to train the model # 
def train(model, ims, real_input_flag, itr):
    cost = model.train(ims, real_input_flag)
    if config['reverse_input']:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % config['display_interval'] == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

# Used to Train the model # 

## TO DO : Complete this ## 

def train_wrapper(model):
    eta = 1.0 # Tunable 
    BATCH_SIZE = config['total_length']

    for itr in range(1, config['max_iterations'] + 1):
        iter = itr % ITER_PER_BATCH
        if iter*BATCH_SIZE >= SIZE_TRAIN_DATA: 
            print('Train Data end. Restarting from first frame')
            eta = 1.0 

        batch = train_data[iter*BATCH_SIZE:(itr+1)*BATCH_SIZE] # Get training batch 

        print(batch[:,1].shape)
        break 

        if config['reverse_scheduled_sampling'] == 1:
            real_input_flag = reserve_schedule_sampling_exp(iter)
        else:
            eta, real_input_flag = schedule_sampling(eta, iter)

        # train(model, batch, real_input_flag, iter) # train model on batch 

        # if iter % config['snapshot_interval'] == 0:
        #     model.save(iter)

        # if iter % config['test_interval'] == 0:
        #     test(model, config, itr) # Test after some training 



##############################################################################################################

# Convinience Training Functions # 
def reserve_schedule_sampling_exp(itr):
    if itr < config['r_sampling_step_1']:
        r_eta = 0.5
    elif itr < config['r_sampling_step_2']:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - config['r_sampling_step_1']) / config['r_exp_alpha'])
    else:
        r_eta = 1.0

    if itr < config['r_sampling_step_1']:
        eta = 0.5
    elif itr < config['r_sampling_step_2']:
        eta = 0.5 - (0.5 / (config['r_sampling_step_2'] - config['r_sampling_step_1'])) * (itr - config['r_sampling_step_1'])
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (config['batch_size'], config['input_length'] - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (config['batch_size'], config['total_length'] - config['input_length'] - 1))
    true_token = (random_flip < eta)

    ones = np.ones((config['img_width'] // config['patch_size'],
                    config['img_width'] // config['patch_size'],
                    config['patch_size'] ** 2 * config['img_channel']))
    zeros = np.zeros((config['img_width'] // config['patch_size'],
                      config['img_width'] // config['patch_size'],
                      config['patch_size'] ** 2 * config['img_channel']))

    real_input_flag = []
    for i in range(config['batch_size']):
        for j in range(config['total_length'] - 2):
            if j < config['input_length'] - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (config['input_length'] - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (config['batch_size'],
                                  config['total_length'] - 2,
                                  config['img_width'] // config['patch_size'],
                                  config['img_width'] // config['patch_size'],
                                  config['patch_size'] ** 2 * config['img_channel']))
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros((config['batch_size'],
                      config['total_length'] - config['input_length'] - 1,
                      config['img_width'] // config['patch_size'],
                      config['img_width'] // config['patch_size'],
                      config['patch_size'] ** 2 * config['img_channel']))
    
    if itr < 50000:
        eta -= 0.00002
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (config['batch_size'], config['total_length'] - config['input_length'] - 1))
    true_token = (random_flip < eta)
    ones = np.ones((config['img_width'] // config['patch_size'],
                    config['img_width'] // config['patch_size'],
                    config['patch_size'] ** 2 * config['img_channel']))
    zeros = np.zeros((config['img_width'] // config['patch_size'],
                      config['img_width'] // config['patch_size'],
                      config['patch_size'] ** 2 * config['img_channel']))
    real_input_flag = []
    for i in range(config['batch_size']):
        for j in range(config['total_length'] - config['input_length'] - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (config['batch_size'],
                                  config['total_length'] - config['input_length'] - 1,
                                  config['img_width'] // config['patch_size'],
                                  config['img_width'] // config['patch_size'],
                                  config['patch_size'] ** 2 * config['img_channel']))
    return eta, real_input_flag


# For testing model # 

## TO DO: Remove test_input_handle to custom logic for our dataset ## 
def test(model, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    res_path = os.getcwd() + "/viscous_fingering"
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs['total_length'] - configs['input_length']):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs['reverse_scheduled_sampling'] == 1:
        mask_input = 1
    else:
        mask_input = configs['input_length']

    real_input_flag = np.zeros(
        (configs['batch_size'],
         configs['total_length'] - mask_input - 1,
         configs['img_width'] // configs['patch_size'],
         configs['img_width'] // configs['patch_size'],
         configs['patch_size'] ** 2 * configs['img_channel']))

    if configs['reverse_scheduled_sampling'] == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    itr = 0 
    BATCH_SIZE = config['batch_size']
    while (itr*BATCH_SIZE < len(test_data)):
        batch_id = batch_id + 1
        test_ims = test_data[itr*BATCH_SIZE:(itr+1)*BATCH_SIZE]
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs['img_channel']]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs['total_length'] - configs['input_length'] 
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs['input_length'], :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs['batch_size'], 3, configs['img_width'], configs['img_width']])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs['batch_size'], 3, configs['img_width'], configs['img_width']])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs['batch_size']):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True,channel_axis=-1)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs['total_length']):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs['input_length']) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs['batch_size'])
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs['total_length'] - configs['input_length']):
        print(img_mse[i] / (batch_id * configs['batch_size']))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs['total_length'] - configs['input_length']):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs['total_length'] - configs['input_length']):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs['total_length'] - configs['input_length']):
        print(lp[i])


train_wrapper(model)