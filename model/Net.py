import math
import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import numpy as np

from model.LSTM import LSTM

logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        self.params = params

        self.embedding_item = nn.Embedding(params.num_emb_item, params.dim_emb_item)
        self.embedding_dept = nn.Embedding(params.num_emb_dept, params.dim_emb_dept)
        self.embedding_cat = nn.Embedding(params.num_emb_cat, params.dim_emb_cat)
        self.embedding_store = nn.Embedding(params.num_emb_store, params.dim_emb_store)
        self.embedding_state = nn.Embedding(params.num_emb_state, params.dim_emb_state)

        self.input_size = params.input_lstm_dim



        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=params.lstm_hidden_size,
            num_layers=params.lstm_num_layers,
            dropouti=params.dropouti,
            dropouto=params.dropouto,
            dropoutw=params.dropoutw,
            batch_first=False  
        )

        self.distribution_preMu = nn.Linear(params.lstm_hidden_size * params.lstm_num_layers, 1)
        self.distribution_preA = nn.Linear(params.lstm_hidden_size * params.lstm_num_layers, 1)
        
        self.distribution_mu = nn.Softplus ()
        self.distribution_a = nn.Softplus()
        

    def forward(self, x_input, stat_cov, hidden, cell):
        
        seq_len, batch_size, _ = x_input.shape
        
        emb_item = self.embedding_item(stat_cov[:, 0].long())
        emb_dept = self.embedding_dept(stat_cov[:, 1].long())
        emb_cat = self.embedding_cat(stat_cov[:, 2].long())
        emb_store = self.embedding_store(stat_cov[:, 3].long())
        emb_state = self.embedding_state(stat_cov[:, 4].long())
        

        emb_concat = torch.cat((emb_item, emb_dept, emb_cat, emb_store, emb_state), dim=1).unsqueeze(0).repeat (seq_len, 1 , 1)


        lstm_input = torch.cat((x_input, emb_concat), dim=2)
        
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        hidden_perm = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        pre_a = self.distribution_preA(hidden_perm)
        pre_mu = self.distribution_preMu (hidden_perm)
        a = self.distribution_a(pre_a).squeeze(-1)
        mu = self.distribution_mu(pre_mu).squeeze(-1)

        return mu, a, hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_num_layers, input_size,
                           self.params.lstm_hidden_size, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_num_layers, input_size,
                           self.params.lstm_hidden_size, device=self.params.device)

    def test(self, x, v_batch, hidden, static_cov, cell):

        seq_len, batch_size, features = x.shape
        num_samples = self.params.sample_size
        predict_steps = self.params.predict_steps
        device = self.params.device

        x_expanded = x.repeat_interleave(num_samples, dim=1)
    
    
        decoder_hidden = hidden.repeat_interleave(num_samples, dim=1)
        decoder_cell = cell.repeat_interleave(num_samples, dim=1)
    
        static_cov_expanded = static_cov.repeat_interleave(num_samples, dim=0)
        v_batch_expanded = v_batch.repeat_interleave(num_samples, dim=0)

        preds = torch.zeros(predict_steps, batch_size * num_samples, device=device)

        pred = None


        for t in range(predict_steps):
            timestep = self.params.start_predict + t

            if timestep < seq_len:
                input_t = x_expanded[timestep].unsqueeze(0)
            else:
                input_t = torch.zeros(1, batch_size * num_samples, features, device=device)
                if pred is not None:
                    input_t[0, :, 0] = preds[t-1] / v_batch_expanded[:, 0]

            de_mu, de_a, decoder_hidden, decoder_cell = self(
                input_t, static_cov_expanded, decoder_hidden, decoder_cell
            )
            
            de_mu = v_batch_expanded[:, 0] * de_mu + v_batch_expanded [:, 1]
            de_a = de_a / torch.sqrt(v_batch_expanded[:, 0]) 
                        
            eps = 1e-8
            r = 1.0 / (de_a + eps)
            logits = torch.log(de_a * de_mu + eps)

            NB_distrib = torch.distributions.NegativeBinomial(r , logits)
            
            preds[t] = NB_distrib.sample().squeeze()
            pred = preds[t]
            
            if t < predict_steps - 1 and timestep + 1 < seq_len and t > 0:
                x_expanded[timestep + 1, :, 0] = pred

        preds_reshaped = preds.view(predict_steps, batch_size, num_samples)


        samples = preds_reshaped.permute(2, 1, 0)

        sample_mediam = torch.median(preds_reshaped, dim=2)[0].transpose(0, 1)
        sample_mu = torch.mean(preds_reshaped, dim=2)[0].transpose(0, 1)
        
        

        return samples, sample_mu, sample_mediam


def loss_fn(mu: torch.Tensor, a: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    eps = 1e-8
    r = 1.0 / (a + eps)
    logits = torch.log(a * mu + eps)
    
    distribution_NB = torch.distributions.negative_binomial.NegativeBinomial(r, logits=logits)
    
    nll = -distribution_NB.log_prob(labels)
    
    masked_nll = nll * mask
    

    valid_days = torch.sum(mask)

    return torch.sum(masked_nll) / torch.clamp(valid_days, min=1.0)

def accuracy_ND_(median: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, relative=False):
    
    null = torch.zeros_like(labels)

    masked_mu = median * mask
    masked_lab = labels * mask
    
    diff = torch.sum(torch.abs(masked_mu - masked_lab), dim = 1) / (torch.sum (mask, dim = 1) + 1e-8)
    if relative:
        summ = torch.ones(labels.shape[1], device=labels.device, dtype=labels.dtype)
    else:
        summ = torch.sum(torch.abs(masked_lab), dim = 1)
    
    return diff.detach().cpu(), summ.detach().cpu()



def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, relative=False):
    diff = torch.sum((mu[mask] - labels[mask]) ** 2).item()

    if relative:
        return [diff, torch.sum(mask).item(), torch.sum(mask).item()]
    else:
        summation = torch.sum(torch.abs(labels[mask])).item()
        if summation == 0:
            logger.error('summation denominator error!')
        return np.array ([diff, summation, torch.sum(mask).item()])

def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor,  mask: torch.Tensor, relative=False):
    
    null = torch.zeros_like(labels)

    masked_mu = torch.where (mask, mu, null)
    masked_lab = torch.where (mask, labels, null)
    
    diff = torch.sum( (masked_mu - masked_lab) ** 2, dim = 1) 

    if relative:
        return [diff, torch.sum(mask, dim = 1), torch.sum(mask, dim = 1)]
    else:
        summation = torch.sum(labels * mask, dim = 1)
        return diff.detach().cpu(), summation.detach().cpu(), torch.sum(mask, dim = 1).detach().cpu()


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, relative=False):
    pred_samples = samples.shape[0]
    
    min_sample_rou = torch.quantile(samples, q = rou, dim=0)

    abs_diff = labels - min_sample_rou 

    loss =  2 * torch.max (rou * abs_diff, (rou - 1) * abs_diff)
    
    numerator = torch.sum (loss * mask.float()).item()
    
    
    if relative:
        denominator = torch.sum(mask).item()                       
    else:
        denominator = torch.sum(torch.abs (labels[mask])).item()
    
    return np.array ([numerator, denominator])   
    
    
def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, relative=False):

    min_sample_rou = torch.quantile(samples, q = rou, dim=0) 

    abs_diff = (labels - min_sample_rou) 
    
    
    loss =  2 * torch.max (rou * abs_diff, (rou - 1) * abs_diff)
    numerator = torch.where (mask, loss, torch.zeros_like(loss)).sum (dim = 1) 
                            
    if relative: 
        denominator = torch.sum(mask, dim = 1)
    else:
        denominator = torch.sum(torch.abs (labels * mask.float()), dim = 1)
    
    return numerator.detach().cpu(), denominator.detach().cpu()

def quantile_CRPS (samples: torch.Tensor, labels: torch.Tensor, quantile_grid, mask: torch.Tensor, relative = False):

    quantile_rize = len(quantile_grid)
    
    quantile_samples = torch.quantile (samples, q = quantile_grid, dim = 0)
    
    pred_quantile_grid = quantile_grid.view (-1, 1, 1)
    
    diff_labels = labels.unsqueeze(0) 
    

    coef_norm = 2 / quantile_rize
    
    diff = diff_labels - quantile_samples
    loss = torch.max(pred_quantile_grid * diff, (pred_quantile_grid - 1.0) * diff)
    
    loss = loss * mask.unsqueeze(0)
    
    numerator = coef_norm * torch.sum (loss).item()
                            
    if relative:
        denominator = torch.sum(mask).item()                       
    else:
        denominator = torch.sum(torch.abs (labels[mask]) ).item()
                            
    return np.array ([numerator, denominator])



def quantile_CRPS_(samples: torch.Tensor, labels: torch.Tensor, quantile_grid, mask: torch.Tensor,
                   relative: bool = False):

    if not isinstance(quantile_grid, torch.Tensor):
        quantile_grid = torch.tensor(quantile_grid, dtype=torch.float32, device=samples.device)
    else:
        quantile_grid = quantile_grid.to(device=samples.device, dtype=torch.float32)

    quantile_size = len(quantile_grid)

    quantile_samples = torch.quantile(samples, q=quantile_grid, dim=0)

    pred_quantile_grid = quantile_grid.view(-1, 1, 1)

    diff_labels = labels.unsqueeze(0)

    diff = diff_labels - quantile_samples
    loss = torch.max(pred_quantile_grid * diff, (pred_quantile_grid - 1.0) * diff)

    loss = loss * mask.unsqueeze(0)

    coef_norm = 2.0 / quantile_size

    numerator = coef_norm * torch.sum(loss)

    if relative:
        denominator = torch.sum(mask)
    else:
        denominator = torch.sum(torch.abs(labels * mask))
        
    denominator = torch.clamp(denominator, min=1e-8)

    return numerator.detach().cpu(), denominator.detach().cpu()
