"""
__author__ = "Mohammed Aldosari"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def transform_data(self):
    self.data
    if self.normalization == 'z-score':
        self.scalers = {}
        for i in range(0, self.n_features):
            scaled = StandardScaler()
            tr_size = int(math.ceil(self.data.shape[0] * self.training_ratio))
            if self.skip_insample is None:
                scaled.fit(self.data.iloc[0:tr_size, i].values.reshape(-1, 1))
            else:
                scaled.fit(self.data.values.reshape(-1, 1))
            transformed = scaled.transform(self.data.iloc[:, i].values.reshape(-1, 1))
            self.scalers['scaled' + self.data.columns[i]] = scaled
            self.data.iloc[:, i] = pd.DataFrame(transformed)
        for idx, (scaler_name, scaler) in enumerate(self.scalers.items()):
            mean = scaler.mean_
            std = scaler.scale_
            #print('-------------------check function for "S".===========================')
            print(f"{self.columns[idx]} Train Mean: {mean[0]:.3f}, Train Standard Deviation: {std[0]:.3f}")
    elif self.normalization == 'minmax':
        self.scalers = {}
        self.min_ = 0
        self.max_ = 1
        for i in range(0, self.n_features):
            scaler = MinMaxScaler(feature_range=(self.min_, self.max_))
            scaler.fit(self.data.iloc[0:self.train_size, i].values.reshape(-1, 1))
            rescaled = scaler.transform(self.data.iloc[:, i].values.reshape(-1, 1))
            self.scalers['scaler_' + self.data.columns[i]] = scaler
            self.data.iloc[:, i] = pd.DataFrame(rescaled)
    elif self.normalization == 'log':
        self.scalers = {}
        if self.features == 's':
            self.shift_values = {self.data.columns[self.data.columns.get_loc(self.target.lower())]: abs(self.data[self.target.lower()].min()) if self.data[self.target.lower()].min() < 0 else 0}
        else:
            self.shift_values = {col: abs(self.data[col].min()) if self.data[col].min() < 0 else 0 for col in self.data.columns}
        self.shift_df = pd.DataFrame([self.shift_values], columns=self.data.columns)
        
        self.data[self.data.columns] += self.shift_df.iloc[0, 0:]
        self.data = np.log1p(self.data)        
                
    return self.data

def inverse_transformation(self):
    if self.normalization == 'min_max':
        mins = np.zeros(len(self.scalers))
        maxs = np.zeros(len(self.scalers))
        for idx, (scaler_name, scaler) in enumerate(self.scalers.items()):
            mean = scaler.data_min_
            std = scaler.data_max_
            mins[idx] = mean
            maxs[idx] = std
        mins = mins.reshape(1, -1, 1)
        maxs = max.reshape(1, -1, 1)
        forecasts = self.forecast_tensor * (maxs - mins) + mins
    elif self.normalization == 'z-score':
        means = np.zeros(len(self.scalers))
        stds = np.zeros(len(self.scalers))

        for idx, (scaler_name, scaler) in enumerate(self.scalers.items()):
            mean = scaler.mean_
            std = scaler.scale_
            means[idx] = mean
            stds[idx] = std
        if self.features == 'S':
            features = 1
        else:
            features = self.n_features
        means = means.reshape(1, 1, features)
        stds = stds.reshape(1, 1, features)
        forecasts = self.forecast_tensor * stds + means
    elif self.normalization == 'log':
        if np.any(self.forecast_tensor < 0) == True:
            raise ValueError(
                f"self.forecast_tensor contains {np.sum(self.forecast_tensor < 0)} negative numbers. inverse_log transformaion will result in NaN.\n"
                f"Please either change the normalization or do further debugging."
            )
        
        forecasts = np.zeros(self.forecast_tensor.shape)
        observed_data = self.data[-forecasts.shape[0]:]
        for k in range(self.forecast_tensor.shape[1]):  
            residuals = observed_data.values - self.forecast_tensor[:,k,:]
            residual_variance = np.nanvar(residuals)
            bias_correction = residual_variance / 2
            forecasts[:, k, :] = np.expm1(self.forecast_tensor[:, k,:])   
        
    else:
        forecasts = self.forecast_tensor
    return forecasts