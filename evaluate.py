import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.Net as net
from dataloader import *

import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training') 

# переделать отображение графиков 
def plot_eight_windows(plot_dir,
                       predict_values,
                       labels,
                       p_10, 
                       p_90, 
                       p_25, 
                       p_75,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], p_10[m], p_90[m], color='blue', alpha=0.15, label='80% CI (P10-P90)')
        
        ax[k].fill_between(x[predict_start:], p_25[m], p_75[m], color='blue', alpha=0.30, label='50% CI (P25-P75)')
                
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')




        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}' \
                                 f'CRPS: {plot_metrics["CRPS"][m]: .3f}'   

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()

import numpy as np
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def evaluate(model, test_loader, params, plot_num, sample=True):
    model.eval()
    torch.cuda.empty_cache()

    plot_batch = 0
    raw_metrics = utils.init_metrics(sample=sample)
    device = params.device

    with torch.no_grad():
        for i, (test_batch, stat_cov, v, labels, mask_labels) in enumerate(tqdm(test_loader)):

            test_batch = test_batch.permute(1, 0, 2).to(device, dtype=torch.float32, non_blocking=True)
            stat_cov = stat_cov.to(device, non_blocking=True)
            v_batch = v.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)

            batch_size = test_batch.shape[1]
            start_predict = params.start_predict


            history_input = test_batch[:start_predict]
            mu_hist, a_hist, hidden, cell = model(history_input, stat_cov)

            v_scale = v_batch[:, 0].unsqueeze(1)  

            input_mu = (v_scale * mu_hist.permute(1, 0)).contiguous()
            input_a = (a_hist.permute(1, 0) / torch.sqrt(v_scale + 1e-8)).contiguous()


            samples, sample_mu, sample_median = model.test(test_batch, v_batch, hidden, stat_cov, cell)

            raw_metrics = utils.update_metrics(
                raw_metrics, input_mu, input_a, sample_mu, sample_median, 
                labels, start_predict, samples, mask_labels, relative=params.relative_metrics
            )

            if i == plot_batch:
                metrics_args = (sample_mu, sample_median, labels, start_predict)
                if sample:
                    sample_metrics = utils.get_metrics(*metrics_args, samples, mask_labels, relative=params.relative_metrics)
                else:
                    sample_metrics = utils.get_metrics(*metrics_args, mask_labels, relative=params.relative_metrics)

                CRPS_scores = np.array(sample_metrics['CRPS'][0])
                
                top_10_count = max(1, batch_size // 10)
                top_10_SRPS_sample = np.argsort(-CRPS_scores)[:top_10_count]

                mask = np.ones(batch_size, dtype=bool)
                mask[top_10_SRPS_sample] = False
                not_chosen = np.where(mask)[0]

                random_sample_10 = np.random.choice(top_10_SRPS_sample, size=min(10, len(top_10_SRPS_sample)), replace=False)
                random_sample_90 = np.random.choice(not_chosen, size=min(10, len(not_chosen)), replace=False)
                combined_sample = np.concatenate((random_sample_10, random_sample_90))

                label_plot = labels[combined_sample].cpu().numpy()
                predict_mu = sample_mu[:, combined_sample, :].cpu().numpy()
                predict_samples = samples[:, combined_sample, :].cpu().numpy()

                plot_mu = np.concatenate((input_mu[combined_sample].cpu().numpy(), predict_mu), axis=1)
                
                p_10 = np.quantile (predict_samples, q = 0.1, axis = 0)
                p_90 = np.quantile (predict_samples, q = 0.9, axis = 0)
                p_25 = np.quantile (predict_samples, q = 0.25, axis = 0)
                p_75 = np.quantile (predict_samples, q = 0.75, axis = 0)
                
                plot_metrics = {_k: _v[0][combined_sample] for _k, _v in sample_metrics.items()}
                
                plot_eight_windows(
                    params.plot_dir, plot_mu, label_plot, p_10, p_90, p_25, p_75,
                    params.train_windows, start_predict, plot_num, plot_metrics, sample
                )

    summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
    metrics_string = '; '.join(f'{k}: {v:05.3f}' for k, v in summary_metric.items())
    logger.info('- Full test metrics: ' + metrics_string)

    return summary_metric

# if __name__ == '__main__':
#     # Load the parameters
#     args = parser.parse_args()
#     model_dir = os.path.join('experiments', args.model_name) 
#     json_path = os.path.join(model_dir, 'params.json')
    
    
    
#     test_dir = os.path.join(args.data_folder, args.dataset_test)
    
#     assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
#     params = utils.Params(json_path)

#     utils.set_logger(os.path.join(model_dir, 'eval.log'))

#     params.relative_metrics = args.relative_metrics
#     params.sampling = args.sampling
#     params.model_dir = model_dir
#     params.plot_dir = os.path.join(model_dir, 'figures')
    
#     cuda_exist = torch.cuda.is_available()  # use GPU is available

#     # Set random seeds for reproducible experiments if necessary
#     if cuda_exist:
#         params.device = torch.device('cuda')
#         logger.info('Using Cuda...')
#         model = net.Net(params).cuda()
#     else:
#         params.device = torch.device('cpu')
#         # torch.manual_seed(230)
#         logger.info('Not using cuda...')
#         model = net.Net(params)

#     # Create the input data pipeline
#     logger.info('Loading the datasets...')

    

#     test_set = TestDataset(test_dir)
#     test_loader = DataLoader(test_set, batch_size=params.batch_size, sampler=RandomSampler(test_set), num_workers=4, pin_memory=True)
   
#     logger.info('- done.')

#     print('model: ', model)
#     loss_fn = net.loss_fn

#     logger.info('Starting evaluation')

#     utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

#     test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
#     save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
#     utils.save_dict_to_json(test_metrics, save_path)