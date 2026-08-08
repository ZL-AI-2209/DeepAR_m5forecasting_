
import json
import logging
import os
import shutil

import torch
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['savefig.dpi'] = 300 #Uncomment for higher plot resolutions
import matplotlib.pyplot as plt

import model.Net as net

logger = logging.getLogger ('DeepAR.Utils')


class Params:

    def __init__(self, json_path):
              
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def set_logger(log_path):
    
    _logger = logging.getLogger('DeepAR')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))

def save_dict_to_json(d, json_path):

    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint (save_dict, is_best, epoch, checkpoint, ins_name=-1):

    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')

    if not os.path.exists(checkpoint):

        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')

        os.mkdir(checkpoint)

    torch.save(save_dict, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')

def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def plot_all_epoch(variable, save_name, location='./figures/'):

    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()

def init_metrics(sample = True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
    }
    
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)

        metrics['CRPS'] = np.zeros(2)
    return metrics


def grid_generation (N_samples):
    return np.arange (0.05, 1, 0.05)

def get_metrics(sample_mu, sample_median, labels, predict_start, mask_labels, samples=None, relative=False):
    metric = dict()

    metric['ND'] = net.accuracy_ND_(sample_median, labels[:, :-predict_start], mask_labels, relative=relative)
    metric['RMSE'] = net.accuracy_RMSE_(sample_mu, labels[:, :-predict_start], mask_labels, relative=relative)
    if samples is not None:   
        grid_crps = grid_generation(samples.shape[0])
        metric['rou90'] = net.accuracy_ROU_(0.9, samples, labels[:, :-predict_start], mask_labels, relative=relative)
        metric['rou50'] = net.accuracy_ROU_(0.5, samples, labels[:, :-predict_start], mask_labels, relative=relative)

        metric['CRPS'] = net.quantile_CRPS_(samples, labels[:, :-predict_start], grid_crps, mask_labels, relative=relative)
    return metric

def update_metrics(raw_metrics, input_mu, input_a, sample_mu, sample_median,  labels, predict_start, mask_labels, samples=None, relative=False):

    
    raw_metrics['ND'] = raw_metrics['ND'] + net.accuracy_ND(sample_median, labels[:, :-predict_start], mask_labels, relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + net.accuracy_RMSE(sample_mu, labels[:, :-predict_start], mask_labels, relative=relative)

    input_time_steps = input_mu.numel()
    loss_value = net.loss_fn(input_mu, input_a, labels[:, -predict_start:], mask_labels)
        
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + np.array ([loss_value.item() * input_time_steps, input_time_steps])
    
    
    if samples is not None:
        grid_crps = grid_generation(samples.shape[0])
        raw_metrics['rou90'] = raw_metrics['rou90'] + net.accuracy_ROU(0.9, samples, labels[:, :-predict_start], mask_labels, relative=relative)
        raw_metrics['rou50'] = raw_metrics['rou50'] + net.accuracy_ROU(0.5, samples, labels[:, :-predict_start], mask_labels, relative=relative)
        raw_metrics['CRPS'] = raw_metrics['CRPS'] + net.quantile_CRPS (samples, labels[:, :-predict_start],  grid_crps, mask_labels, relative=relative)
        
    return raw_metrics


def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
                raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['test_loss'] = (raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]

        summary_metric['CRPS'] = raw_metrics['CRPS'][0]/raw_metrics['CRPS'][1]

    return summary_metric
