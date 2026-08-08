from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
import io

from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
import pandas as pd
import math

import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import kagglehub


from zarr.codecs import BloscCodec



import os
import numpy as np
import zarr
from numcodecs import Blosc as BloscCodec
from tqdm import trange

def prep_data(data, static_cov, dyn_cov, prices, data_start,
              window_size, stride_size, save_path, save_name,
              train=True):

    store_path = os.path.join(save_path, f"{save_name}.zarr")
    root = zarr.open_group(store=store_path, mode='w')

    time_len = data.shape[1]
    num_series = data.shape[0]
    num_covariates = dyn_cov.shape[0]
    num_static = static_cov.shape[1]

    static_cov = np.asarray(static_cov, dtype='int16')
    
    data = np.nan_to_num(np.asarray(data, dtype='float32').T, nan=0.0, posinf=0.0, neginf=0.0)
    dyn_cov = np.nan_to_num(np.asarray(dyn_cov, dtype='float32').T, nan=0.0, posinf=0.0, neginf=0.0)
    prices = np.nan_to_num(np.asarray(prices, dtype='float32').T, nan=0.0, posinf=0.0, neginf=0.0)

    input_size = window_size - stride_size

    if train:
        windows_per_series = (time_len - data_start - input_size) // stride_size
    else:
        windows_per_series = np.ones(num_series, dtype='int32')

    windows_per_series = np.maximum(windows_per_series, 0)
    total_windows = int(np.sum(windows_per_series))
    print(f"Total windows (Train={train}): {total_windows}")

    if total_windows == 0:
        raise ValueError("Нет окон для создания! Проверьте параметры.")

    lag = np.zeros((2, time_len, num_series), dtype='float32')
    lag[0, 30:, :] = data[:-30, :]
    lag[1, 364:, :] = data[:-364, :]

    total_features = 3 + num_covariates + 2
    label_len = window_size 

    # --- УСТАНАВЛИВАЕМ КРУПНЫЙ РАЗМЕР ЧАНКА ---
    # 2048 или 4096 — отличный выбор для стабильности на Windows
    CHUNK_SIZE = 2048 

    compressor_config = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle"
        }
    }

    x_input_arr = root.create_array(
        name='x_input', shape=(total_windows, window_size - 1, total_features),
        chunks=(CHUNK_SIZE, window_size - 1, total_features), dtype='float32', compressors=[compressor_config]
    )
    v_input = root.create_array(
        name='v_input', shape=(total_windows, 2),
        chunks=(CHUNK_SIZE, 2), dtype='float32', compressors=[compressor_config]
    )
    labels_arr = root.create_array(
        name='labels', shape=(total_windows, label_len),
        chunks=(CHUNK_SIZE, label_len), dtype='float32', compressors=[compressor_config]
    )
    stat_cov = root.create_array(
        name='series_id', shape=(total_windows, num_static),
        chunks=(CHUNK_SIZE, num_static), dtype='int16', compressors=[compressor_config]
    )
    
    labels_mask_arr = root.create_array(
        name='labels_mask', shape=(total_windows, label_len),
        chunks=(CHUNK_SIZE, label_len), dtype='float32', compressors=[compressor_config]
    )

    buffer_x, buffer_v, buffer_labels, buffer_stat, buffer_mask = [], [], [], []
    buffered_count = 0
    count = 0

    for series in trange(num_series):
        num_w = windows_per_series[series]
        if num_w == 0:
            continue

        if train:
            start_idx = data_start[series]
            window_starts = np.arange(start_idx, start_idx + num_w * stride_size, stride_size)
            offsets = np.arange(window_size)
            indices = window_starts[:, None] + offsets[None, :]
        else:
            indices = np.arange(time_len - window_size, time_len)[None, :]

        if indices.max() >= time_len:
            print(f"WARNING: series {series}, max index {indices.max()} >= {time_len}")
            continue

        v_input_local = np.ones((num_w, 2), dtype='float32')
        for w in range(num_w):
            window_data = data[indices[w, :input_size], series]
            nonzero_sum = (window_data != 0).sum()
            if nonzero_sum > 0:
                v_input_local[w, 0] = window_data.sum() / nonzero_sum + 1.0

        x_input = np.zeros((num_w, window_size - 1, total_features), dtype='float32')
        v_safe = v_input_local[:, 0:1]


        target_data = data [indices[:, :-1], series].copy()

        x_input[:, :, 0] = target_data / v_safe
        x_input[:, :, 1] = prices[indices[:, :-1], series] != 0
        x_input[:, :, 2] = prices[indices[:, :-1], series] / v_safe
        x_input[:, :, 3:3 + num_covariates] = dyn_cov[indices[:, :-1], :]

        lag_0 = lag[0, indices[:, :-1], series].copy()
        lag_1 = lag[1, indices[:, :-1], series].copy()


        x_input[:, :, 3 + num_covariates] = lag_0 / v_safe
        x_input[:, :, 4 + num_covariates] = lag_1 / v_safe  
        
        
        labels_mask = prices[indices, series] != 0
        labels = data[indices, series].astype('float32')


        # Добавляем данные в списки ОЗУ
        buffer_x.append(x_input)
        buffer_v.append(v_input_local)
        buffer_labels.append(labels)
        buffer_stat.append(np.broadcast_to(static_cov[series], (num_w, num_static)))
        buffer_mask.append(labels_mask)
        buffered_count += num_w

        # Если накопили достаточно для полного чанка (или больше) — пишем одним махом
        if buffered_count >= CHUNK_SIZE:
            bx = np.concatenate(buffer_x, axis=0)
            bv = np.concatenate(buffer_v, axis=0)
            bl = np.concatenate(buffer_labels, axis=0)
            bs = np.concatenate(buffer_stat, axis=0)
            bm = np.concatenate(buffer_mask, axis=0)

            x_input_arr[count:count + buffered_count] = bx
            v_input[count:count + buffered_count] = bv
            labels_arr[count:count + buffered_count] = bl
            stat_cov[count:count + buffered_count] = bs
            labels_mask_arr[count:count + buffered_count] = bm

            count += buffered_count
            buffer_x, buffer_v, buffer_labels, buffer_stat, buffer_mask = [], [], [], []
            buffered_count = 0

    # Сбрасываем остатки из буфера, которые не заполнили целый чанк
    if buffered_count > 0:
        bx = np.concatenate(buffer_x, axis=0)
        bv = np.concatenate(buffer_v, axis=0)
        bl = np.concatenate(buffer_labels, axis=0)
        bs = np.concatenate(buffer_stat, axis=0)
        bm = np.concatenate(buffer_mask, axis = 0 )

        x_input_arr[count:count + buffered_count] = bx
        v_input[count:count + buffered_count] = bv
        labels_arr[count:count + buffered_count] = bl
        stat_cov[count:count + buffered_count] = bs
        labels_mask_arr[count:count + buffered_count] = bm
        count += buffered_count

    print(f"Данные сохранены в Zarr. Всего записей: {count}")
    return root

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()


def pred_price(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    print("start pred price")

    prices_wide = prices.pivot(
        index=['item_id', 'store_id'],
        columns='wm_yr_wk',
        values='sell_price'
    ).reset_index()

    d_to_wm = calendar.set_index('d')['wm_yr_wk']
    d_to_date = calendar.set_index('d')['date']

    date_cols = [c for c in sales.columns if c.startswith('d_')]
    id_info = sales[['id', 'item_id', 'store_id']]

    merged = id_info.merge(prices_wide, on=['item_id', 'store_id'], how='left')
    merged = merged.set_index('id')

    wm_cols = [d_to_wm[d] for d in date_cols]
    result = merged[wm_cols].copy()
    result.columns = date_cols

    result.columns = result.columns.map(d_to_date.get)

    print("finish pred price")
    return result

def pred_dyn_cov(calendar: pd.DataFrame) -> pd.DataFrame:
    print ("start dyn cov")
    cov_id = ['weekday_sin', 'weekday_cos', 'wday', 'month_sin', 
              'month_cos', 'year_encoder', 'event_type_1_encoder']
    num_dyn_cov = len(cov_id)


    covariates = pd.DataFrame(index=range(num_dyn_cov + 1), columns=calendar['date'].unique())

    calendar_copy = calendar.copy()
    
    calendar_copy['weekday_encoder'] = OrdinalEncoder().fit_transform(
        calendar_copy[['weekday']]
    ).flatten()
    
    calendar_copy['year_encoder'] = OrdinalEncoder().fit_transform(
        calendar_copy[['year']]
    ).flatten()
    
    calendar_copy['event_type_1'] = calendar_copy['event_type_1'].fillna('no_event')
    calendar_copy['event_type_1_encoder'] = OrdinalEncoder().fit_transform(
        calendar_copy[['event_type_1']]
    ).flatten()
    
    # Циклические признаки
    T_week = 7
    calendar_copy['weekday_sin'] = np.sin(
        calendar_copy['weekday_encoder'] * math.pi * 2 / T_week
    )
    calendar_copy['weekday_cos'] = np.cos(
        calendar_copy['weekday_encoder'] * math.pi * 2 / T_week
    )
    
    T_month = 12
    calendar_copy['month_sin'] = np.sin(
        calendar_copy['month'] * math.pi * 2 / T_month
    )
    calendar_copy['month_cos'] = np.cos(
        calendar_copy['month'] * math.pi * 2 / T_month
    )

    covariates.iloc[0, :] = 0
    for idx, cov in enumerate(cov_id):
        covariates.iloc[idx + 1, :] = (
            calendar_copy.groupby('date')[cov].first().values
        )
    print ("finish dyn cov ")
    return covariates

def pred_static_cov (sales: pd.DataFrame) -> pd.DataFrame:
    print("start stat cov ")
    encoded_cols = []
    id_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for cov in id_vars: 
        le = LabelEncoder()
        encoded_col = pd.Series(
            le.fit_transform(sales[cov]),
            name=f'{cov}_enc',
            index=sales.index
        )
        encoded_cols.append(encoded_col)
    sales = pd.concat([sales] + encoded_cols, axis=1)
    
    static_covs = pd.DataFrame(sales[[f'{c}_enc' for c in id_vars]].values, columns = id_vars)
    print("finish stat cov ")
    return static_covs

def mask_generate()

if __name__ == '__main__':


    params_obj = utils.Params('experiments/base_model/params.json')
    params = params_obj.dict


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

    data = sales.loc[:, 'd_1':'d_1941'].values
    days = data.shape[1]

    calendar = calendar[(calendar['date'] >= train_start) & (calendar['date'] <= test_end)].reset_index(drop=True)
    num_series = data.shape[0]


    data_frame = pd.DataFrame(data, columns= calendar['date'])

    static_cov = pred_static_cov(sales)

    dyn_cov = pred_dyn_cov(calendar)


    prices_data = pred_price(sales, calendar, prices)


        
    train_data = data_frame.loc[:, train_start:train_end].to_numpy()
    train_prices = prices_data.loc[:, train_start:train_end].to_numpy()


    test_data = data_frame.loc[:, test_start:test_end].to_numpy()
    test_prices = prices_data.loc[:, test_start:test_end].to_numpy()

    train_cov = dyn_cov.loc[:, train_start:train_end].to_numpy()
    test_cov = dyn_cov.loc[:, test_start:test_end].to_numpy()
    
    data_start = (data!=0).argmax(axis=1)


    prep_data(train_data, static_cov, train_cov, train_prices, data_start, window_size=window_size,
                     stride_size=stride_size, save_path = save_path, save_name = save_name_test, train=True)
    
    prep_data(test_data, static_cov, test_cov, test_prices, data_start, window_size=window_size,
                    stride_size=stride_size, save_path = save_path, save_name = save_name_test, train=False) 

