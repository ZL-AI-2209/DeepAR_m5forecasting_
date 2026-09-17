import pandas as pd
import numpy as np
import os
import zipfile
import io
import kagglehub
import matplotlib.pyplot as plt
import utils
from scipy.stats import chi2
from pathlib import Path

import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning

# Отключаем конкретное предупреждение об обращении матрицы Гессе
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler

import faiss
from prophet import Prophet

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import CubicSpline

from tqdm import trange
from tqdm import tqdm

from scipy import stats 
import statsmodels.api as sm
import pymannkendall as mk #Статистика Панна - Кендала

# реализация cusum алгоритма для исключения OOS нулей https://habr.com/ru/companies/magnit/articles/918928/
# синтетический контроль (для проверки тренда в условиях ограниченности выборки) https://habr.com/ru/companies/skillfactory/articles/700260/


import matplotlib

import preprocess_elect


matplotlib.use('TkAgg') 

def Score_test_type_series(data, alpha):
    
    poisson_mask = np.zeros(data.shape[0])
    N = data.shape[1]
    eps = 0.0001
    
    number_null = (data == 0).sum(axis=1)
    
    p_null_empirical = number_null/ N
    
    few_zeros = p_null_empirical < 0.05
    poisson_mask [few_zeros] = True
    
    test_score_index = (~few_zeros)
    
    sample_mean = data[test_score_index].sum(axis = 1) / N 
    
    exp_mean = np.exp(- sample_mean)
    
    numerator = (number_null [test_score_index] - N  * exp_mean) ** 2
    
    denumerator = N * exp_mean * ( 1 -exp_mean - sample_mean * exp_mean)    
    
    S = numerator / (denumerator + eps )
    
    p_value = 1.0 - chi2.cdf(S, df=1)
    
    poisson_mask [test_score_index] = (p_value > alpha)
    ZIP_mask = ~poisson_mask.astype(bool)
    
    
    if (ZIP_mask.any):
        pi_hat = np.clip((p_null_empirical [ ZIP_mask & test_score_index ] - exp_mean[ ZIP_mask [test_score_index ] ]) 
                     / np.maximum(1e-9, 1 - exp_mean [ZIP_mask [test_score_index ]] ), 0.0, 0.99)
    
    return poisson_mask.astype(bool), pi_hat

def h_threshold_search_poisson (window_size, lambda_0, N_samples, error_rate):
    num_channels = lambda_0.shape[0]
    H = np.zeros_like(lambda_0)
    G = np.zeros(N_samples)
    
    eps = 0.0001
    lambda_1 = 0.05
    
    log_lambda_ratio = np.log(lambda_1 / (lambda_0 + eps) )
    lambda_diff = lambda_1 - lambda_0
    
    G_max = np.zeros_like (N_samples)
    
    for i, lamb in enumerate(lambda_0):
        sample_poisson = np.random.poisson(lam=lamb, size=(N_samples, window_size))
        
        s = sample_poisson * log_lambda_ratio [i] - lambda_diff[i] 
        
        for k in range (window_size):
            G_k = np.maximum(0, G + s[:, k])
            G_max = np.maximum (G_k, G_max)
        
        H[i] = np.quantile(G_max, q = 1 - error_rate )
    
    return H 

def zero_inflated_poisson(lam, p_zero, size=1000):

    is_zero = np.random.binomial(1, p_zero, size=size)

    poisson_vals = np.random.poisson(lam, size=size)
    return np.where(is_zero == 1, 0, poisson_vals)

def h_threshold_search_ZIP (window_size, lambda_0, p_value, N_samples, error_rate):
    
    num_channels = lambda_0.shape[0]
    H = np.zeros( (num_channels, p_value.shape[0]) )
    G = np.zeros(N_samples)
    
    eps = 0.0001
    lambda_1 = 0.05
    
    
    log_lambda_ratio = np.log(lambda_1 / (lambda_0 + eps) )
    lambda_diff = lambda_1 - lambda_0
    p1_zero = p_value + (1 - p_value) * np.exp(-lambda_1)
    
    G_max = np.zeros_like (N_samples)
    
    for i, lamb in enumerate(lambda_0):

        p0_zero = p_value + (1 - p_value) * np.exp(-lamb)
        
        for r, p in enumerate(p_value):
            sample_ZIP = zero_inflated_poisson(lam=lamb, p_zero = p, size=(N_samples, window_size))
            
            null_ZIP = (sample_ZIP == 0)
            not_null_ZIP = ~null_ZIP
            
            s = np.zeros ((N_samples, window_size))
            
            s[not_null_ZIP] = sample_ZIP [not_null_ZIP] * log_lambda_ratio [i] - lambda_diff[i] 
            
            s[null_ZIP] = np.log(np.maximum(p1_zero[r], 1e-9) / np.maximum(p0_zero[r], 1e-9)) 
    
            
            for k in range (window_size):
            
                G_k = np.maximum(0, G + s[:, k])
                
                G_max = np.maximum (G_k, G_max)
        
            H[i, r] = np.quantile(G_max, q = 1 -  error_rate )
    
    return H 


def cusum_detected(window, decomposed_window, mask_window, poisson_series, pi_hat, pre_threshold_poisson, pre_threshold_ZIP):

    num_channels = window.shape[0]
    window_size = window.shape[1]
    eps = 0.0001
    
    N = mask_window.sum(axis = 1)
    lambda_0 = ((decomposed_window).sum(axis = 1) / N) + eps 
    lambda_1 = 0.05

    ZIP_series = ~poisson_series

    zip_indices = np.where(ZIP_series)[0]

    S = np.zeros((num_channels, window_size))
    G = np.zeros((num_channels, window_size))
    h = np.zeros(num_channels)
    OOS_start = np.zeros(num_channels)
    OOS_end = np.zeros(num_channels)
    triggered_channels = np.zeros(num_channels)
    
    p0_zero = pi_hat + (1 - pi_hat) * np.exp(-lambda_0[ZIP_series])
    p1_zero = pi_hat + (1 - pi_hat) * np.exp(-lambda_1)
    
    log_lambda_ratio = np.log(lambda_1 / lambda_0)
    lambda_diff = lambda_1 - lambda_0   
    zip_s_zero = np.log(np.maximum(p1_zero, 1e-9) / np.maximum(p0_zero, 1e-9))
    
    h[poisson_series] = pre_threshold_poisson(lambda_0[poisson_series])
    h[ZIP_series] = pre_threshold_ZIP(lambda_0[ZIP_series], pi_hat, grid=False)
    
    for k in tqdm(range(window_size), desc="Временные шаги", leave=False):
        s = np.zeros_like(lambda_0)
        x_k = window[:, k]
        
        poisson_series_mask = poisson_series & mask_window[:, k].astype(bool)
        if np.any(poisson_series_mask):
            s[poisson_series_mask] = (x_k[poisson_series_mask] * log_lambda_ratio[poisson_series_mask] 
                                 - lambda_diff[poisson_series_mask])
            
        zip_is_active = mask_window[zip_indices, k].astype(bool)
        zip_is_zero = (x_k[zip_indices] == 0)
        
      
        mask_zero_in_zip = zip_is_active & zip_is_zero
        mask_nonzero_in_zip = zip_is_active & (~zip_is_zero)

        zero_indices_in_s = zip_indices[mask_zero_in_zip]
        nonzero_indices_in_s = zip_indices[mask_nonzero_in_zip]
        
        s[zero_indices_in_s] = zip_s_zero[mask_zero_in_zip]
        s[nonzero_indices_in_s] = (x_k[nonzero_indices_in_s] * log_lambda_ratio[nonzero_indices_in_s] 
                                   - lambda_diff[nonzero_indices_in_s])

        if k > 0:
            S[:, k] = S[:, k - 1] + s
            G[:, k] = np.maximum(0, G[:, k - 1] + s)
        else: 
            G[:, k] = np.maximum(0, s)
            S[:, k] = s 
            
    
    for ch in tqdm(range(num_channels), desc="Анализ каналов", leave=False):
        alarm_steps = np.where(G[ch, :] > h[ch])[0]
        
        if len(alarm_steps) > 0:
            first_alarm = alarm_steps[0]
            zero_steps = np.where(G[ch, :first_alarm] == 0)[0]
            
            if len(zero_steps) > 0: 
                OOS_start[ch] = zero_steps[-1] + 1
                OOS_end[ch] = first_alarm 
                triggered_channels[ch] = 1 

    return OOS_start, OOS_end, triggered_channels
    
def search_max_lambda_p (data, data_start, window_size):
    max_lambda = 0
    min_lambda = 1e+10
            
    max_p_value = 0
    min_p_value = 1.0
    
        
        
    len_series = data.shape[1]
        
    windows_number = (len_series - data_start - window_size)
    
    max_window = np.max (windows_number)
    print ("Подготовка границ для сетки порогов:")
    for i in trange (max_window): 
                
        window_series = np.where ( (data_start + i + window_size) <= len_series - 1) [0]
                    
        for idx in window_series:
            
            window = data[idx][data_start[idx].item() + i: data_start[idx].item() + i + window_size]
            N = window.shape[0]
    
            lambda_0 = window.sum()/ N 
            null_number = (window == 0).sum()
            
            
            p_null_empirical = null_number / N
            exp_mean = np.exp (-lambda_0)

            p_value = np.clip( (p_null_empirical - exp_mean) / np.maximum(1e-9, 1 - exp_mean ) , 0.0, 0.99)
            
            max_lambda = max (lambda_0, max_lambda) 
            min_lambda = min (lambda_0, min_lambda) 
        
            max_p_value = max (p_value, max_p_value) 
            min_p_value = min (p_value, min_p_value)
    
    return max_lambda, min_lambda, min_p_value, max_p_value
    
def pred_lambda_p( min_lambda, max_lambda, min_p_value, max_p_value , window_size, num_points = 50):
    

    lambda_grid = np.exp (np.linspace(np.log(1), np.log (max_lambda), num = num_points - 1))
    lambda_grid = np.insert (lambda_grid, 0, min_lambda)
    
    i = np.arange(num_points)

    p_value_grid = min_p_value + (max_p_value - min_p_value) * np.sin(np.pi * i / (2 * (num_points - 1))) ** 2

    func_poisson = h_threshold_search_poisson (window_size, lambda_grid, 1000, 0.05)    
    spline_poisson = CubicSpline (lambda_grid, func_poisson)
    
    func_ZIP = h_threshold_search_ZIP (window_size, lambda_grid, p_value_grid, 1000, 0.05)
    spline_ZIP = RectBivariateSpline (lambda_grid, p_value_grid, func_ZIP, kx=3, ky=3)
    
    return spline_poisson, spline_ZIP

def prophet_decomposed(data, data_start, calendar, path_save):
    decomposed_series = np.zeros_like (data)

    data_size = data.shape [0]
    len_data = data.shape [1]
    
    weekS_strength = np.zeros (data_size)
    yearS_strength = np.zeros (data_size)
    Zero_Ratio = np.zeros(data_size)
    CV = np.zeros(data_size)
    
    
    holidays_date = pd.to_datetime(calendar[calendar['event_name_1'].notna()].date)
    holidays_name = calendar['event_name_1'].dropna().tolist()
    holidays = pd.DataFrame ({
        'holiday': holidays_name,
        'ds': holidays_date
    })
    
    median_series = np.median (data, axis = 1)
    print ("Старт декомпозиции рядоа...")
    
    for i in trange(data_size):
        
        start = data_start[i]
        dates = calendar['date'][start:].reset_index(drop=True)
        data_serie = data[i][start:]
        
        nonzero_only = np.where( data_serie > 0,  data_serie, np.nan)
        dataFrame = pd.DataFrame ({'y': nonzero_only, 
                           'ds': calendar['date'][start:]})
        
        model = Prophet(
                holidays= holidays,
                yearly_seasonality=True, 
                weekly_seasonality=True, 
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                holidays_mode='multiplicative',
                changepoint_prior_scale=0.05,      
                seasonality_prior_scale=15.0      
            )

        model.fit (dataFrame)
        
        hist_dates = dataFrame[['ds']].copy()
        decomposition = model.predict(hist_dates)

        decomp_df = pd.DataFrame({
            'trend': decomposition ['trend'],
            'yearly': decomposition['yearly'],
            'weekly': decomposition['weekly'],
            'yhat': decomposition['yhat']
        })
        

        denum_factor = np.where(decomp_df ['yhat'] <= 0, 1, decomp_df ['yhat'])

        decomposed_series[i, start:] = data_serie / denum_factor
        
        ers = data_serie - decomp_df ['yhat']
        
        yearly_abs = decomp_df['trend'] * (1 + decomp_df['yearly'])
        weekly_abs = decomp_df['trend'] * (1 + decomp_df['weekly'])
        
        var_ers = np.var (ers)
        
        var_yearly_component = np.var(yearly_abs) + np.var(ers)
        var_weekly_component = np.var(weekly_abs) + np.var (ers)
        
        yearS_strength[i] = np.maximum (0.0, 1 - var_ers / var_yearly_component)
        weekS_strength[i] = np.maximum (0.0, 1 - var_ers / var_weekly_component )
        
        
        N = len_data - start
        nonzero_count = np.count_nonzero(data_serie > 0)

        if nonzero_count > 0:
            Zero_Ratio[i] = 1 -  nonzero_count / N
        else:
            Zero_Ratio[i] = 0.0
              

        std_val = np.std(data_serie)
        mean_val = np.mean(data_serie)

        if mean_val > 0:
            CV[i] = std_val / mean_val
        else:
            CV[i] = 0.0
            

    print ("Декомпозиция закончена")

    np.save(f"{path_save}/decomposed_series.npy", decomposed_series)


    df_stats = pd.DataFrame({
        'weekS_strength': weekS_strength,
        'yearS_strength': yearS_strength,
        'Zero_Ratio': Zero_Ratio,
        'R_CV': CV
    })

    df_stats.to_csv(f"{path_save}/series_features.csv", index=False)
    
    
    print (f"Результаты сохранены: {path_save}")
        
def generate_similarities_series(sales, data, data_start, yearS_strength, weekS_strength,  Zero_Ratio,  CV, path_save, number_similar_series):
    

    rize_data = data.shape[0]
    len_data = data.shape[1]
    
    median_series = np.zeros (rize_data)
    Q_9 = np.zeros (rize_data)
    Q_1 = np.zeros (rize_data)
    Q_75 = np.zeros (rize_data) 
    Q_25 = np.zeros (rize_data)
    
    print ("Подготовка данных аналогов:\n")
    for i in trange (rize_data):
        median_series[i] = np.median(data[i, data_start[i]:])
        Q_9[i] = np.quantile (data[i, data_start[i]:], 0.9)
        Q_1[i] = np.quantile (data[i, data_start[i]:], 0.1)
        Q_75[i] = np.quantile (data[i, data_start[i]:], 0.75) 
        Q_25[i] = np.quantile (data[i, data_start[i]:], 0.25)
    
    
        
    max_median = np.max (median_series)
    
    series_analog = np.zeros((rize_data, number_similar_series), dtype=int)
    

    print("Старт подготовки аналогов...")
    
    iqr = Q_75 - Q_25      
    p_range = Q_9 - Q_1     


    Kelly_Skewness = np.where(
        p_range != 0, 
        (Q_9 - 2 * median_series + Q_1) / p_range, 
        0.0
    )
    
    Coef_Hogg = np.where(
        iqr != 0, 
        p_range / iqr, 
        0.0
    )
    
    relative_KS = Kelly_Skewness / np.max (Kelly_Skewness)
    relative_Hogg = Coef_Hogg / np.max (Coef_Hogg)
    

    len_series = (len_data - data_start) / len_data
    relative_median = median_series / max_median
    relative_CV = CV / np.max(CV) 
    relative_yearS = yearS_strength / np.max (yearS_strength)
    relative_weekS = weekS_strength / np.max (weekS_strength)
    
    x = np.column_stack((
        Zero_Ratio, 
        relative_CV, 
        relative_median, 
        relative_KS,
        relative_Hogg,
        relative_yearS, 
        relative_weekS, 
        len_series
    ))
    
    weight = np.array([2.5, 2.0, 1.75, 1.75 , 1.5, 1.5, 1.0, 0.5])
    
    scaler = RobustScaler()
    feature_vector = scaler.fit_transform(x) * weight
    
    index = faiss.IndexFlatL2(feature_vector.shape[1])
    index.add(feature_vector.astype(np.float32))
    
    dist, index_analog = index.search(feature_vector.astype(np.float32), 150) 
    
    for i in trange(rize_data):
        analog_indices = index_analog[i][1:] 
        
        dist_idx = dist[i][1:].copy()
        
        item_id = sales.iloc[i]['item_id']
        dept_id = sales.iloc[i]['dept_id']
        cat_id = sales.iloc[i]['cat_id']
        store_id = sales.iloc[i]['store_id']
        state_id = sales.iloc[i]['state_id']
        
        analog_sales = sales.iloc[analog_indices]
        
        dept_penalty = 3.0 * (analog_sales['dept_id'] != dept_id).values.astype(float)
        store_penalty = 0.25 * (analog_sales['store_id'] != store_id).values.astype(float)
        state_penalty = 0.5 * (analog_sales['state_id'] != state_id).values.astype(float)
        
        dist_idx += dept_penalty + store_penalty + state_penalty
        
        final_idx = np.argsort(dist_idx)
        series_analog[i] = analog_indices[final_idx][:number_similar_series]
        
    np.save((f"{path_save}/series_analog_{str (number_similar_series)}.npy") , series_analog)
        
    print("Аналоги подготовлены...")


def checking_OOS(sales, data, mask, start, stop, data_start, series_trigger, pre_period, similarities_series_id):
    
    sorted_indices = np.argsort(start[series_trigger])
    sorted_series_trigger = series_trigger[sorted_indices]
    
    print ("    Подтверждение результатов cusum алгоритма ...")
    
    for idx_serie in tqdm  (sorted_series_trigger):
        
        item_id = sales.iloc[idx_serie]['item_id']
        state_id = sales.iloc[idx_serie]['state_id']
        id_ = sales.iloc[idx_serie]['id']
        
        start_cell = start[idx_serie]
        stop_cell = stop[idx_serie]
        
        start_pre_period = start_cell - pre_period
        len_period = stop_cell - start_cell
        
        idex_goods_index = (sales[(sales['item_id'] == item_id) & (sales['id'] != id_)]).index
        id_reg_analog = (sales.loc[idex_goods_index, 'state_id'] == state_id).values
        
        similarities_series = similarities_series_id[idx_serie][:10] 
        
        
        true_start_close_analog = data_start[idex_goods_index] < (start_cell - 45)
        idex_goods_index = idex_goods_index[true_start_close_analog]
        id_reg_analog = id_reg_analog[true_start_close_analog]
        
        true_close_analog = idex_goods_index.size > 0
        
        if (true_close_analog):
        
            max_start_close_analog = np.max(data_start[idex_goods_index])
            max_start_close_analog = max(data_start[idx_serie], max_start_close_analog)
            
            idex_analog = np.concatenate((np.array([idx_serie]), idex_goods_index, similarities_series))    
        else:
            idex_analog = similarities_series
            
        number_analog = idex_analog.size
        number_close_analog = (idex_goods_index.size)
        
       
        number_series = 0
        number_discrete = 0
        number_discrete_close_analog = 0
        discrete_series = []


        while (number_series < number_analog) and (number_discrete < 150 or number_series < number_close_analog + 1):

            
            idx = idex_analog[number_series]
            
            if (number_series <= (number_close_analog) ) & true_close_analog:
                start_slice = max_start_close_analog if (start_cell - max_start_close_analog <= 365) else (start_cell - 365)
                 
            else: 
                start_slice = data_start[idx] if data_start[idx] <= start_cell else start_cell
                start_slice = start_slice if (start_cell - start_slice) <= 365 else (start_cell - 365)
                                
            
            discrete_serie = np.zeros(365)
            mask_series = mask[idx, start_slice:start_cell]
            series = data[idx, start_slice:start_cell]
            not_null_value = np.where((series > 0) & mask_series)[0]
            discrete_serie[:len(not_null_value[1:])] = not_null_value[1:] - not_null_value[:-1] - 1.
            
            if (number_series > 0 or not true_close_analog):
                number_discrete += len(discrete_serie[discrete_serie != 0])
                
            if (number_series > 0 and number_series <= number_close_analog):
                number_discrete_close_analog += len(discrete_serie[discrete_serie != 0])

            number_series += 1

            discrete_series.append(discrete_serie)
        
        
        discrete_series = np.array(discrete_series)
            
        if (true_close_analog):
            cell_discrete = discrete_series[0, discrete_series[0] != 0]
            close_analog_discrete = discrete_series[1:(number_close_analog + 1)]
            analog_discrete = discrete_series[1:]
            
            start_slice = max_start_close_analog if (start_cell - max_start_close_analog <= 90) else start_cell - 90
        
            analog = data[idex_goods_index, start_slice:start_cell]
        
            cell_series = data[idx_serie, start_slice:start_cell]
            mask_cell = mask[idx_serie, start_slice:start_cell].copy()
            mask_analog = mask[idex_goods_index, start_slice:start_cell].copy()
        
        else:
            analog_discrete = discrete_series
            

        filter_1 = False
        filter_2 = False
        filter_3 = False
        filter_4 = False
        
        if number_discrete > 3:
            filter_1 = dynamics_positional_discreteness(analog_discrete, len_period, 0.05)
        
        if true_close_analog and number_discrete_close_analog > 2:
            filter_2 = discreteness_trend(close_analog_discrete, id_reg_analog, cell_discrete)
            filter_3 = sales_trend(analog, mask_analog, id_reg_analog, cell_series, mask_cell)
            filter_4 = min_analog(close_analog_discrete, analog, mask_analog, len_period)
        
        
        if (number_discrete < 3 and number_discrete_close_analog < 2):
            mask_filter = False
        else:
            mask_filter = not (filter_1 | filter_2 | filter_3 | filter_4)
            
        mask[idx_serie, start_cell:stop_cell] = mask_filter # в случае невозможности проверки из за отсутсвия данных принимаем оос 
        
        debug_plot = False
        
    print ("    Результаты проверены. Этап пройден.")
    return mask 

def dynamics_positional_discreteness (discret_array, len_period, alpha):
    
    discret_array = discret_array.flatten()

    discret_array = discret_array [discret_array != 0]
    
    if (discret_array.size == 0): 
        return False
    
    if np.var(discret_array) == 0:
        return False
    
    exog = np.ones_like(discret_array)
    model = sm.NegativeBinomial(discret_array, exog)
    results = model.fit(disp = 0)

    
    mu_fit = np.exp(results.params[0]) 
    alpha_fit = results.scale 
    
    p_scipy = 1 / (1 + alpha_fit * mu_fit)
    n_scipy = 1 / alpha_fit

    prob = stats.nbinom.pmf(len_period, n=n_scipy, p=p_scipy)
    
    return prob < alpha
    
    
def discreteness_trend (discret_analog, id_reg_analog, cell_discretness):
    
    
    rize_analog_pull = discret_analog.shape[0]
    
    if (cell_discretness.size <= 2):
        return False 
    
    result = mk.original_test(cell_discretness)
    cell_slope = result.slope
    
    cell_true_trend = (result.p < 0.05) & (cell_slope < 0)
    
    true_null_array = np.zeros (rize_analog_pull, dtype = bool)
    
    confirm_1 = False
    confirm_2 = False
    
    if (cell_true_trend):
        presence_trend = np.zeros(rize_analog_pull, dtype = bool)
        slope = np.zeros(rize_analog_pull)
    
        for i, series in enumerate (discret_analog):
            series = series [series != 0]
                
            if (series.size <= 2):
                true_null_array[i] = False
                break

            true_null_array[i] = True 
            result = mk.original_test(series)
        
            presence_trend[i] = result.p < 0.05
            slope[i] = result.slope

        presence_trend = presence_trend[true_null_array]
        slope = slope[true_null_array]
        id_reg_analog = id_reg_analog[true_null_array]
        
        if (true_null_array.sum() > 0):
            if presence_trend.sum () > 0.6 * rize_analog_pull:
                median_slope = np.median(slope[presence_trend])
            
                if median_slope < 0 and abs(cell_slope) > 1e-8:
                    confirm_1 = ( median_slope / (cell_slope) ) > 0.5 # 2 медианы
                    
                
            if (presence_trend[id_reg_analog] != 0).all() and (len(slope[id_reg_analog]) > 0): 
                median_slope = np.median (slope[id_reg_analog])
                
            
                if median_slope < 0 and abs(cell_slope) > 1e-8: 
                    confirm_2 = (median_slope / (cell_slope)) > 0.5 

    return confirm_1 | confirm_2
    


def sales_trend (analog, mask, id_reg_analog, cell_series, mask_cell):

    len_series = analog.shape[1]
    t = np.arange(len_series)
        
    reg_analog = ((~mask).sum (axis = 1) / len_series) < 0.5
    
    analog = analog[reg_analog]
    mask = mask[reg_analog]
    id_reg_analog = id_reg_analog[reg_analog]
     
    not_valid_value = ~mask
    
    rize_analog = analog.shape [0]
    
    valid_null_value_cell = (cell_series == 0.0) & mask_cell
    valid_null_value = (analog == 0.0) & mask
    
    null_number_cell = valid_null_value_cell.sum() / mask_cell.sum()
    null_number = np.sum (valid_null_value, axis = 1) / mask.sum (axis = 1)
    
    mask_cell_regression = np.ones_like(cell_series)
    mask_cell_regression[~mask_cell] = 0.0
    

    many_null = null_number > 0.7

    mask_regression = np.ones_like (analog)
    mask_regression [not_valid_value] = 0.0
    
    number_valid_analog = reg_analog.sum()
    
    reg_true = np.zeros (number_valid_analog, dtype = bool)
    coef_reg = np.zeros(number_valid_analog)

    if many_null.sum() / number_valid_analog > 0.5:
        mask_cell_regression[valid_null_value_cell] = (1 - null_number_cell)
        for i in range (number_valid_analog):
            mask_regression[i, valid_null_value[i]] = (1 - null_number[i])
            
    WLS_model_cell = sm.WLS (cell_series, sm.add_constant(t), weights = mask_cell_regression)
    result_cell = WLS_model_cell.fit()
    
    reg_true_cell = result_cell.pvalues[1] < 0.05 #проверка значимости оценки тренда  
    
    conf_1 = False
    conf_2 = False
    cell_slope = result_cell.params[1]
    
    if reg_true_cell and (result_cell.params[1] < 0.0):
    
        for i in range (number_valid_analog):
            WLS_model = sm.WLS (analog[i], sm.add_constant(t), weights = mask_regression[i])
            
            res_regression = WLS_model.fit ()
            reg_true [i] = res_regression.pvalues[1] < 0.05
            coef_reg[i] = res_regression.params[1]
            
        if (reg_true.sum()/number_valid_analog) > 0.7 and len (coef_reg[reg_true]) > 0:
            median_coef = np.median (coef_reg[reg_true])
            
        
            if (median_coef < 0.0 ) and abs(cell_slope) > -1e-8:
                conf_1 = (median_coef / cell_slope) > 0.5
            
                
        if ((reg_true[id_reg_analog] != 0.0).all() and len (reg_true[id_reg_analog]) > 0):
            median_coef = np.median (coef_reg[id_reg_analog])
            
                    
            if (median_coef < 0.0) and abs(cell_slope) > 1e-8:
                conf_2 = (median_coef / cell_slope) > 0.5                        

    return conf_1 | conf_2

def min_analog (discret_array, analog, mask_analog, len_period):
    
    mean_analog = analog.sum(axis = 1) / (mask_analog.sum (axis = 1) + 0.01)
    
    id_min = np.argsort(mean_analog)
    
    for i in range(len(id_min)):
        id_min_analog = id_min[i]
        not_null_min_analog = discret_array[id_min_analog][discret_array[id_min_analog] != 0]
        
        if len(not_null_min_analog) > 0:
            break
    
    median_discret = np.median (not_null_min_analog)
    
    return len_period > 2.0 * median_discret

    
def generate_mask (data, sales, calendar, start_number_win, stride, window_size, data_start, path_save, number_similar_series):
    
    len_series = data.shape[1]
    data_size = data.shape[0]
    
    path_mask = Path((f"{path_save}/mask.npy"))
    if path_mask.is_file():
        mask = np.load (f"{path_save}/mask.npy")
        print (f"Загрузка маски: старт с окна {start_number_win}")
    else: 
        mask = np.ones_like (data).astype(np.bool)
        start_number_win = 0
        print (f"Генерация маски")

    windows_number = (len_series - data_start - window_size + 1 ) // stride
    
    max_window = np.max (windows_number)
    
    path_decom = Path((f"{path_save}/decomposed_series.npy"))

    if path_decom.is_file():
        decomposed_series = np.load (f"{path_save}/decomposed_series.npy")
        df_stats = pd.read_csv (f"{path_save}/series_features.csv")
    else: 
        prophet_decomposed(data, data_start, calendar, path_save= path_save)
        decomposed_series = pd.read_csv (path_decom)

    path_analog = Path((f"{path_save}/series_analog_{str (number_similar_series)}.npy"))
    if path_analog.is_file():
        series_analog = np.load (path_analog)
    else: 
        generate_similarities_series(
            sales, data, data_start, 
            df_stats ['yearS_strength'],df_stats ['weekS_strength'], df_stats ['Zero_Ratio'], df_stats ['R_CV'], path_save, 
            number_similar_series = number_similar_series
        )
        series_analog = np.load (f"{path_save}/series_analog_{str (number_similar_series)}.npy")

    #max_lambda, min_lambda, min_p_value, max_p_value = search_max_lambda_p (decomposed_series, data_start, window_size)
    
    #print (max_lambda, min_lambda, min_p_value, max_p_value )

    spline_poisson, spline_ZIP = pred_lambda_p(min_lambda = 0.0, 
                                               max_lambda = 155.74479166666666,
                                               min_p_value = 0.0, 
                                                max_p_value = 0.99, 
                                                window_size = window_size, 
                                                num_points = 50)
    
    poisson_mask, pi = Score_test_type_series(data, 0.05)
    
    print ("Начало формирования маски...")    

    number_null = (data == 0).sum () - data_start.sum ()

    for i in range (start_number_win, max_window):
        
        print (f"Этап {i}/{max_window}. Начало cusum алгоритма.")
        start_OOS_window = np.zeros(data_size)
        end_OOS_window = np.zeros(data_size)

        triggered_channels = np.zeros(data_size)
        
        window_start_series = data_start + i*stride

        len_mask_window = (window_start_series + window_size) <= len_series
        window_series = np.where ( len_mask_window) [0]
        len_iter_data = len (window_series)
        
        data_windows = np.zeros ((len_iter_data, window_size))
        mask_windwos = np.zeros ((len_iter_data, window_size))
        data_decomposed = np.zeros ((len_iter_data, window_size))
        
        poisson_mask_iter = poisson_mask [window_series]
        pi_iter = pi [len_mask_window [~poisson_mask]]
        
        for idx, r in enumerate (window_series):
            window_start = window_start_series[r]
            
            data_windows [idx] = data[r][window_start: window_start + window_size]
            mask_windwos [idx] = mask[r][window_start:window_start + window_size]
            data_decomposed [idx] = decomposed_series[r][window_start:window_start + window_size]
            
        start_OOS_window[window_series], end_OOS_window [window_series], triggered_channels [window_series] \
            = cusum_detected (data_windows, data_decomposed, mask_windwos, poisson_mask_iter, pi_iter, spline_poisson, spline_ZIP) 
        
        series_trigger = np.where (triggered_channels)[0]
        start_OOS_window [series_trigger] = window_start_series[series_trigger] + start_OOS_window[series_trigger] 
        end_OOS_window [series_trigger] = window_start_series[series_trigger] + end_OOS_window[series_trigger]
        
        mask = checking_OOS (sales, data, mask, start_OOS_window.astype (int), end_OOS_window.astype (int), data_start, series_trigger, pre_period = 30, similarities_series_id = series_analog)
        # invalid_oos_but_positive = (data > 0) & (~mask)

        # mask[invalid_oos_but_positive] = True
        
        np.save(f"{path_save}/mask.npy", mask)
        print (f"Этап {i} сохранен")
        number_OOS_null = (~mask).sum ()
        
        print(f"Количество OOS нулей относительно обычных: {number_OOS_null / number_null}" )
        print(f"Количество OOS нулей: {number_OOS_null}" )
    print (f"Маска сформирована\nДанные сохранены: {path_save}/mask.npy")

    

    return mask

if __name__ == '__main__':

    params_obj = utils.Params('experiments/base_model/params.json')
    params = params_obj.dict

    path =  'C:/Users/FuckHacker/Desktop/DeepAR-M5forecasting-main/x13as_ascii-v1-1-b62/x13as'


    save_path = params ['save_path']
    save_name_train = params ['save_name_train']
    save_name_test = params ['save_name_test']

    window_size = params ['train_windows']
    stride_size = params['stride_size']
    pred_days = params['predict_steps']

    train_start = params ['train_start']
    train_end = params ["train_end"]

    test_start = params ['test_start']
    test_end = params ['test_end']

    name_zip = params ['name_zip_data']


    if not (os.path.exists(name_zip)):
        name_zip = kagglehub.competition_download('m5-forecasting-accuracy')
    

    with zipfile.ZipFile(name_zip, 'r') as zfile:
        sales = pd.read_csv(io.BytesIO(zfile.read('sales_train_evaluation.csv')))
        calendar = pd.read_csv(io.BytesIO(zfile.read('calendar.csv')))
        prices = pd.read_csv(io.BytesIO(zfile.read('sell_prices.csv')))

    calendar = calendar[(calendar['date'] >= train_start) & (calendar['date'] <= test_end)].reset_index(drop=True)
    data = sales.loc[:, 'd_1':'d_1941'].values

    data_start = (data != 0).argmax(axis=1)
    num_series = data.shape[0]

    path_save = params['path_save_mask_generator']

    generate_mask (data, sales, calendar, 119, 3, window_size, data_start, path_save, 50)
    # есть примерно 0.02 шанс ложных срабатываний, желательно отсматривать ряды с большим количество неверных срабатываний отдельно 
