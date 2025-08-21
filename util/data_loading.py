"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.timefeatures import time_features
import warnings
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from statsmodels.tsa.ar_model import AutoReg
warnings.filterwarnings('ignore')
import numpy as np

def clip_dates(start_date: str, end_date: str, data: pd.DataFrame):
    data['date'] = pd.to_datetime(data['date'])
    start_date = start_date
    end_date   = end_date
    data = data[~((data['date'] >= start_date) & (data['date'] <= end_date))]
    data = data.reset_index(drop=True)
    return data

def load_data(self) -> pd.DataFrame:

    data = pd.read_csv(self.data_path)
    if self.args.clip_dates:
        data = clip_dates(self.args.start_date, self.args.end_date, data)
        
    self.dates = pd.to_datetime(data['date']).dt.date
    self.frequency = None #to be implemented
    data.reset_index(inplace=True, drop=True)
    self.data = data.select_dtypes(include=['number'])
    self.original_shape = self.data.shape

class Dataset(Dataset):
    def __init__(self, experiment, args, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', timeenc=0, freq='h', cycle=None):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.normalization = experiment.normalization
        self.timeenc = timeenc
        self.freq = freq

        self.experiment = experiment
        
        self.dataset = experiment.args['dataset']
        self.clip_dates = experiment.args['clip_dates']
        self.start_date =  experiment.args['start_date']
        self.end_date = experiment.args['end_date']
        
        self.cycle = cycle

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        
        if self.clip_dates:
            df_raw = clip_dates(self.start_date, self.end_date, df_raw)

        self.dates = pd.to_datetime(df_raw['date']).dt.date
        frequencies = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'A': 'Yearly', 'W-MON': 'Weekly',
                       'W-TUE': 'Weekly'}
        self.frequency = frequencies.get(pd.infer_freq(self.dates))

        if self.dataset == 'ETTh1':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24 + (self.pred_len - 1)]
            self.train_size = 12 * 30 * 24 + 4 * 30 * 24
            self.test_size = 0
            self.data_ = df_raw.select_dtypes(include=['number'])
            self.data_ = self.data_.iloc[0:12 * 30 * 24 + 8 * 30 * 24,]
            self.columns = self.data_.columns
        else:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            self.train_size = num_train + num_vali
            self.test_size = num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw) + (self.pred_len - 1)]
            self.data_ = df_raw.select_dtypes(include=['number'])
            self.columns = self.data_.columns

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        zeros_rows = pd.DataFrame(np.zeros(((self.pred_len - 1), len(df_raw.columns))), columns=df_raw.columns)
        df_raw = pd.concat([df_raw, zeros_rows])

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        self.sample_mean = df_data[border1s[0]:border2s[0]].mean().to_frame().T


        if self.normalization == 'z-score':
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_o = np.vstack([(df_data.iloc[:-(self.pred_len - 1)  or None]), df_data.iloc[-(self.pred_len - 1) or None:]])
            data = np.vstack([self.scaler.transform(df_data.iloc[:-(self.pred_len - 1)  or None]), df_data.iloc[-(self.pred_len - 1) or None:]])
            self.data = self.scaler.transform(df_data.iloc[:-(self.pred_len - 1) or None])
            self.data = pd.DataFrame(self.data, columns=train_data.columns)
            self.sample_mean_normalized = self.data[border1s[0]:border2s[0]].mean().to_frame().T
        elif self.normalization == 'log':   
            if self.features == 'S':
                self.shift_values = {df_raw.columns[df_raw.columns.get_loc(self.target)]: abs(df_raw[self.target].min()) if df_raw[self.target].min() < 0 else 0}
            else:
                self.shift_values = {col: abs(df_raw[col].min()) if df_raw[col].min() < 0 else 0 for col in df_raw.columns[1:]}
            
            self.shift_df = pd.DataFrame([self.shift_values], columns=df_data.columns)
            
            df_data[df_data.columns] += self.shift_df.iloc[0, 0:]
            
            data = np.vstack([np.log1p(df_data.iloc[:-(self.pred_len - 1)  or None]), df_data.iloc[-(self.pred_len - 1)  or None:]])
            self.data = np.log1p(df_data.iloc[:-(self.pred_len - 1)  or None])
            self.data = pd.DataFrame(self.data, columns=df_data.columns)
            self.sample_mean_normalized = self.data[border1s[0]:border2s[0]].mean().to_frame().T
        else:
            data = df_data.values
            self.trend_short_term = data
            self.trend_long_term = data
            self.data = df_data.iloc[:-(self.pred_len - 1) or None]
            self.sample_mean_normalized = self.data[border1s[0]:border2s[0]].mean().to_frame().T
            
        if self.features == 'M':
            self.sample_mean = self.sample_mean
            self.sample_mean_normalized = self.sample_mean_normalized
        elif self.features == 'MS':
            self.sample_mean = self.sample_mean.iloc[:, self.columns.get_loc(self.target)]
            self.sample_mean_normalized = self.sample_mean_normalized.iloc[:, self.columns.get_loc(self.target)]
        elif self.features == 'S':
            self.sample_mean = self.sample_mean
            self.sample_mean_normalized = self.sample_mean_normalized
            
            
        if self.dataset == 'ETTh1':
            self.data = self.data.iloc[0:12 * 30 * 24 + 8 * 30 * 24,]
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format='%m/%d/%Y %H:%M', errors='coerce')
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
                        
        indices = np.arange(data.shape[0]).reshape(-1, 1)
        
        self.indices_x = indices[border1:border2]
        self.indices_y = indices[border1:border2]
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]        
        self.data_stamp = data_stamp
        
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        indices_x = self.indices_x[s_begin:s_end]
        indices_y = self.indices_y[r_begin:r_end]
        
        seq_x = self.data_x[s_begin:s_end]
        
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        cycle_index = torch.tensor(self.cycle_index[s_end])
            
        return indices_x, indices_y, seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.normalization == 'z-score':
            forecasts = self.scaler.inverse_transform(data)
        elif self.normalization == 'log':
            forecasts = np.expm1(data) - pd.Series(self.shift_values).values
        else:
            forecasts = data
        return forecasts

def data_provider(experiment, args, flag):
    timeenc = 0 if args['embed'] != 'timeF' else 1
    experiment = experiment

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
        freq = args['freq']
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args['batch_size']
        freq = args['freq']

    data_set = Dataset(
        experiment=experiment,
        args=args,
        data_path=args['data_path'],
        flag=flag,
        size=[args['seq_len'], args['label_len'], args['pred_len']],
        features=args['features'],
        target=args['target'],
        timeenc=timeenc,
        freq=freq,
        cycle=args['cycle'])
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args['num_workers'],
        drop_last=drop_last,
    )
    return data_set, data_loader
