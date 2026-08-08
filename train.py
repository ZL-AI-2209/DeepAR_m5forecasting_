import argparse
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.Net as net
from evaluate import evaluate


import matplotlib


matplotlib.use('Agg')

from dataloader import *

logger = logging.getLogger('DeepAR.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_train', default='elect', help='Name of the dataset')
parser.add_argument('--dataset_test', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')
parser.add_argument('--search-params', default='learning_rate,batch_size',
                    help='Comma-separated list of params to log')




import torch
import numpy as np
from tqdm import tqdm

def train_epoch(model, optimizer, loss_fn, train_loader, params, epoch, scaler=None):

    model.train()
    loss_epoch = np.zeros(len(train_loader))
    device = params.device
    train_windows = params.train_windows


    for i, (train_batch, static_cov, v_batch, labels_batch, mask) in enumerate(tqdm(train_loader)): # оптимизировать
        optimizer.zero_grad(set_to_none=True)

        train_batch = train_batch.permute(1, 0, 2).to(device, dtype=torch.float32, non_blocking=True)
        static_cov = static_cov.to(device, dtype=torch.int64, non_blocking=True)
        labels_batch = labels_batch.permute(1, 0).to(device, dtype=torch.float32, non_blocking=True)
        v_batch = v_batch.to(device, dtype=torch.float32, non_blocking=True)
        mask = mask.permute(1, 0).to(device, dtype=torch.float32, non_blocking=True)

        batch_size = train_batch.shape[1]

        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)


        input_series = train_batch.clone()

        mu, a, hidden, cell = model(input_series, static_cov, hidden, cell)
    
            
        mu = v_batch[:, 0].unsqueeze(0) * mu  + v_batch[:, 1].unsqueeze(0)
        a = a / torch.sqrt(v_batch[:, 0].unsqueeze(0)) 

        total_loss = loss_fn(mu, a, labels_batch[1:], mask)

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        loss_epoch[i] = loss_val

        if i % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {i}/{len(train_loader)}, train_loss: {loss_val:.4f}')

    return loss_epoch


def train_and_evaluate(model, train_loader, test_loader, optimizer, loss_fn,
                       params, restore_file=None, dataset='elect'):
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    logger.info('begin training and evaluation')
    best_test_CRPS = float('inf')
    patience = params.patience
    patience_counter = 0 
    
    train_len = len(train_loader)

    CRPS_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))

    print_params = 'default'

    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))

        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train_epoch(
            model, optimizer, loss_fn, train_loader, params, epoch
        )


        test_metrics = evaluate(model, test_loader, params, epoch, sample=args.sampling)

        CRPS_summary[epoch] = test_metrics['CRPS']
        is_best = CRPS_summary[epoch] <= best_test_CRPS

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict()
        }, epoch=epoch, is_best=is_best, checkpoint=params.model_dir)


            
        if is_best:
            best_test_CRPS = CRPS_summary[epoch]
            patience_counter = 0  
            logger.info('- Found new best CRPS')
            
            if args.save_best:
                list_of_params = args.search_params.split(',')
                
                log_strings = [f'{p}: {getattr(params, p)}' for p in list_of_params]
                text_to_save = ', '.join(log_strings)
                

                file_strings = [f'{p}_{getattr(params, p)}' for p in list_of_params]
                print_params = '_'.join(file_strings)  

               
                with open('./param_search.txt', 'w') as f: 
                    f.write('-----------\n')
                    f.write(text_to_save + '\n')
                    f.write(f'Best CRPS: {best_test_CRPS}\n')
                
                logger.info(text_to_save)
                logger.info(f'Best CRPS: {best_test_CRPS}')
        else:
            patience_counter += 1 
            logger.info(f'No improvement for {patience_counter}/{patience} epochs')
            
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                logger.info(f'Best CRPS: {best_test_CRPS} at epoch {epoch+1-patience_counter}')
                break
            
        

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

    


        utils.plot_all_epoch(CRPS_summary, print_params + '_CRPS', location=params.plot_dir)
        utils.plot_all_epoch(loss_summary, print_params + '_loss', location=params.plot_dir)


if __name__ == '__main__':

    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')

    test_dir = os.path.join(args.data_folder, args.dataset_test)
    train_dir = os.path.join(args.data_folder, args.dataset_train)

    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'

    params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    cuda_exist = torch.cuda.is_available()

    if cuda_exist:
        params.device = torch.device('cuda')
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = net.Net(params)

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info('Loading the datasets...')

    # Инициализируем датасеты по их индивидуальным путям
    train_set = LoadDataset(train_dir)
    test_set = LoadDataset(test_dir)

    # Исправлено: передаем в сэмплер только один аргумент train_dir
    sampler = WeightedSampler(train_dir)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=params.batch_size_test, sampler=RandomSampler(test_set), num_workers=4, pin_memory=True)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    loss_fn = net.loss_fn

    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))

    train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       loss_fn,
                       params,
                       args.restore_file)