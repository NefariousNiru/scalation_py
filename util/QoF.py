"""
__author__ = "Mohammed Aldosari"
__date__ = 2/22/24
__version__ = "1.0"
__listicense__ = "MIT style license file"
"""

from typing import Tuple
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from util.tools import dotdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

"""
QoF: 
    A module for evaulating the quality/goodness of fit
    Supports both symmetric and asymmetric metrics

Functions
----------
smape: 
    def smape(y: np.ndarray, yp: np.ndarray) -> float:

mae:
    def mae(y: np.ndarray, yp: np.ndarray) -> float:

sst:
    def sst(y: np.ndarray) - > float:

sse:
    def sse(y: np.ndarray, yp: np.ndarray) -> float:

r2:
    def r2(y: np.ndarray, yp: np.ndarray) -> float:

mse:
    def mse(y: np.ndarray, yp: np.ndarray) -> float:

rmse:
    def rmse(y: np.ndarray, yp: np.ndarray) -> float:
"""


def smape(self, y: np.ndarray, yp: np.ndarray) -> float:
    """
    A function to calculate the symmetric mean absolute percentage error (sMAPE).

    Arguments
    ----------
    y: np.ndarray
        the observed data
    yp: np.ndarray
        the corresponding forecasts

    Returned Values
    ----------
    smape : float
    """
    epsilon = 1e-8
    numerator = (np.abs(y - yp))
    denominator = (np.abs(y) + np.abs(yp) + epsilon)
    smapes = np.mean(200 * (numerator / denominator), axis = 0)
    smapes = smapes.flatten()
    return smapes


def mae(self, y: np.ndarray, yp: np.ndarray) -> float:
    """
    A function to calculate the mean absolute error (MAE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    yp: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    mae : float
    """
    maes = np.abs(y - yp)
    maes = np.mean(maes, axis = 0)
    if self.args.get('internal_diagnose'):
        return maes
    else:
        maes = maes.flatten()
        return maes


def sst(self, y: np.ndarray, sample_mean) -> float:
    """
    A function to calculate the sum of squares total (SST).

    Arguments
    ----------
    y: np.ndarray
        the response data

    Returned Values
    ----------
    sst : float
    """
    ssts = (y - np.squeeze(sample_mean.values)) ** 2
    ssts = np.sum(ssts, axis = 0)
    return ssts


def sse(self, y: np.ndarray, yp: np.ndarray) -> float:
    """
    A function to calculate the sum of squared errors (SSE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    yp: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    sse : float
    """

    sses = (y - yp) ** 2
    sses = np.sum(sses, axis = 0)
    return sses


def r2q(self, y: np.ndarray, yp: np.ndarray, sample_mean) -> float:
    """
    A function to calculate r squared.

    Arguments
    ----------
    y: np.ndarray
        the response data
    yp: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    r2 : float
    """
    sse_ = sse(self, y, yp)
    sst_ = sst(self, y, sample_mean)

    r2 = 1 - (sse_ / sst_)
    r2 = r2.flatten()
    return r2


def mse(self, y: np.ndarray, yp: np.ndarray) -> float:
    """
    A function to calculate mean squared error (MSE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    yp: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    mse : float
    """
    mses = (y - yp) ** 2
    mses = np.mean(mses, axis = 0)
    mses = mses.flatten()
    return mses


def rmse(self, y: np.ndarray, yp: np.ndarray) -> float:
    """
    A function to calculate root mean squared error (RMSE).

    Arguments
    ----------
    y: np.ndarray
        the response data
    yp: np.ndarray
        the predicted/forecasted outputs

    Returned Values
    ----------
    rmse : float
    """
    rmses = np.sqrt(mse(self, y, yp))
    rmses = rmses.flatten()
    return rmses

def corr(self, y: np.ndarray, yp: np.ndarray) -> float:
    # add docstring
    mean_y = np.mean(y, axis=0)
    mean_yp = np.mean(yp, axis=0)

    covariance = np.mean((y - mean_y) * (yp - mean_yp), axis=0)

    std_y = np.std(y, axis=0)
    std_yp = np.std(yp, axis=0)
    correlations = covariance / (std_y * std_yp + 1e-8)

    return correlations

def bias(self, y: np.ndarray, yp: np.ndarray) -> float:
    # add docstring
    biases = np.mean(yp - y, axis=0)
    return biases

def mase(self, y: np.ndarray, yp: np.ndarray, maes_naive = None) -> float:
    # add docstring
    maes = np.abs(y - yp)
    maes = np.mean(maes, axis = 0)
    maes = maes.flatten()
    if self.args.get('internal_diagnose'):
        mases = maes/maes
    else:
        mases = maes / maes_naive

    return mases

def mape(self, y: np.ndarray, yp: np.ndarray) -> float:
    """
    A function to calculate the mean absolute percentage error (MAPE).

    Arguments
    ----------
    y: np.ndarray
        The observed data
    yp: np.ndarray
        The corresponding forecasts

    Returned Values
    ----------
    mape : float
    """
    epsilon = 1e-8
    mape_values = np.mean(100 * np.abs((y - yp) / (y + epsilon)), axis=0)
    return mape_values

def get_metrics(self, actual, forecasts, sample_mean, maes_naive = None) -> Tuple[int, float, float, float]:
    # add docstring

    if forecasts.ndim == 1:
        valid_indices = ~np.isnan(forecasts)
    else:
        valid_indices = ~np.isnan(forecasts).any(axis=1)

    y = actual[valid_indices]
    yp = forecasts[valid_indices]

    mse_ = mse(self, y, yp)
    rmse_ = rmse(self, y, yp)
    mae_ = mae(self, y, yp)
    smape_ = smape(self, y, yp)
    r2q_ = r2q(self, y, yp, sample_mean)
    sse_ = sse(self, y, yp)
    sst_ = sst(self, y, sample_mean)

    corr_ = corr(self, y, yp)

    bias_ = bias(self, y, yp)

    mase_ = mase(self, y, yp, maes_naive)

    yp_shape = yp.shape
    
    return yp_shape[0], mse_, rmse_, mae_, smape_, mase_, sse_, sst_, r2q_, corr_, bias_

def diagnose(self):
    if self.args.get('mase_calc') is None:
        from modeling.statmodeling.random_walk import RandomWalk
        args = dotdict()
        args.data_path = self.args.get('data_path')
        args.training_ratio = self.args.get('training_ratio')
        args.normalization = self.args.get('normalization')
        args.dataset = self.args.get('dataset')
        args.target = self.args.get('target')
        args.forecast_type = self.args.get('forecast_type')
        args.plot_scope_scale = None
        args.plot_eda = False
        args.qof_calculation_mode = self.args.get('qof_calculation_mode')
        args.horizons = self.horizons
        args.same_n_samples = self.args.get('same_n_samples')
        args.skip_insample = self.args.get('skip_insample')
        args.features = self.args.get('features')
        args.modeling_approach = self.args.get('modeling_approach')
        args.debugging = False
        self.args['mase_calc'] = 'done'
        args.mase_calc = 'done'
        args.internal_diagonse = True
        rw_model = RandomWalk(args)
        self.mae_normalized_list, self.mae_original_list = rw_model.trainNtest()
    
    if self.qof is None:
        self.qof = pd.DataFrame(
            columns=[
                'h',
                'n',
                "MSE Normalized", "RMSE Normalized", "MAE Normalized", "sMAPE Normalized", "MASE Normalized",
                "SSE Normalized", "SST Normalized", "R Squared Normalized", "Corr Normalized", "Bias Normalized",
                "MSE Original", "RMSE Original", "MAE Original", "sMAPE Original", "MASE Original",
                "SSE Original", "SST Original", "R Squared Original", "Corr Original", "Bias Original"
            ]
        )

    self.qof_metrics = {
        col: np.full((self.pred_len, self.forecast_tensor.shape[-1] if self.features == 'm' else 1), np.nan)
        for col in self.qof.columns if col != 'h'
    }

    for h in range(self.pred_len):
        if self.features == 'm':
            forecast_tensor = self.forecast_tensor[:, h, :]
            forecast_tensor_original = self.forecast_tensor_original[:, h, :]
            actual = self.data.iloc[self.train_size:, :].values
            actual_original = self.data_.iloc[self.train_size:, :].values
        elif self.features == 'ms':
            forecast_tensor = self.forecast_tensor[:, h, self.target_feature]
            forecast_tensor_original = self.forecast_tensor_original[:, h, self.target_feature]
            actual = self.data.iloc[self.train_size:, self.target_feature].values
            actual_original = self.data_.iloc[self.train_size:, self.target_feature].values
        elif self.features == 's':
            target_feature = -1
            forecast_tensor = self.forecast_tensor[:, h, target_feature]
            forecast_tensor_original = self.forecast_tensor_original[:, h, target_feature]
            actual = self.data.iloc[self.train_size:, target_feature].values
            actual_original = self.data_.iloc[self.train_size:, self.target_feature].values
        
        if self.args.get('internal_diagnose'):
            normalized_metrics = get_metrics(self, actual, forecast_tensor, self.sample_mean_normalized)
            original_metrics = get_metrics(self, actual_original, forecast_tensor_original, self.sample_mean)
        else:
            if len(self.mae_normalized_list) == 1:
                h = -1
            normalized_metrics = get_metrics(
                self, actual, forecast_tensor, self.sample_mean_normalized, self.mae_normalized_list[h])
            original_metrics = get_metrics(
                self, actual_original, forecast_tensor_original, self.sample_mean, self.mae_original_list[h])

        all_metrics = normalized_metrics + original_metrics[1:]
        qof_keys = [col for col in self.qof.columns if col != 'h']

        for key, val in zip(qof_keys, all_metrics):
            self.qof_metrics[key][h] = val

    for h in self.horizons:
        row = {'h': h}
        if self.qof_calculation_mode == 'single_horizon':
            for col in self.qof.columns:
                if col == 'h':
                    continue
                data = self.qof_metrics[col][h - 1]
                row[col] = int(np.nansum(data)) if col == 'n' else np.nanmean(data)
        elif self.qof_calculation_mode == 'aggregated_horizons':
            for col in self.qof.columns:
                if col == 'h':
                    continue
                data = self.qof_metrics[col][0:h].flatten()
                row[col] = int(np.nansum(data)) if col == 'n' else np.nanmean(data)
        if self.modeling_approach == 'joint':
            self.qof = pd.concat([self.qof, pd.DataFrame([row])], ignore_index=True)
    if self.modeling_approach == 'individual':
        self.qof = pd.concat([self.qof, pd.DataFrame([row])], ignore_index=True)

    if self.args.get('internal_diagnose'):
        return self.qof_metrics['MAE Normalized'], self.qof_metrics['MAE Original']
    else:
        for metric in self.args['qof_to_display'][2:]:

            custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

            plt.figure(figsize=(4.5, 3))

            for i in range(self.qof_metrics[metric].shape[-1]):
                plt.plot(
                    np.arange(1, self.qof_metrics[metric].shape[0] + 1),
                    self.qof_metrics[metric][:, i],
                    label=self.columns[i] if self.features == 'm' else self.columns[self.target_feature],
                    color=custom_colors[i % len(custom_colors)],
                    linewidth=0.5, marker='o', markersize=1.2
                )

            #plt.title(metric, fontsize=10)
            plt.xlabel('Horizons', fontsize=9)
            plt.ylabel(metric, fontsize=9)

            plt.legend(
                fontsize=7.5,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.48),  
                ncol=3,                      
                frameon=False
            )

            plt.tight_layout(rect=[0, 0.05, 1, 1])  
            plt.grid(True, linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            plt.show()