"""
__author__ = "Mohammed Aldosari"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import time
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from util.QoF import diagnose
from util.data_transforms import inverse_transformation
from modeling.statmodeling.model import Model
from util.plotting import plot_forecasts
from statsmodels.tsa.arima.model import ARIMA as ARIMA_
from numpy import ndarray
from util.tools import display_model_info

class ARIMA(Model):
    def __init__(self, args):
        self.p = args['p']
        self.d = args['d']
        self.q = args['q']
        self.rc = args['rc']
        self.trend = args['trend']
        self.fit_method = args['fit_method']
        self.model_name = f'ARIMA({self.p},{self.d},{self.q}) rc = {self.rc}'
        super().__init__(args)

    def train_test(self) -> None:
        self.forecast_tensor: ndarray[float] = np.full(shape=(self.test_size, self.pred_len, self.n_features),
                                                    fill_value=np.nan)

        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0

        for i in tqdm(range(self.n_features)):
            for j in tqdm(range(self.test_size - sample_offset)):
                if j % self.rc ==0:
                    if self.skip_insample is None:
                        train_data = self.data.iloc[0:self.train_size + j, i]
                    else:
                        train_data = self.data.iloc[:, i]
                    arima_model = ARIMA_(train_data, order=(self.p,
                                                           self.d,
                                                           self.q),
                                         trend=self.trend).fit(method = self.fit_method)
                else:
                    if self.skip_insample is None:
                        new_data = self.data.iloc[self.train_size + j - 1:self.train_size + j, i]
                        train_data = pd.concat([train_data, new_data])
                        arima_model = arima_model.append(endog=new_data, refit=False)
                start = self.train_size + j
                end = start + self.pred_len - 1

                forecasts = arima_model.predict(start = start, end = end, dynamic = True).values.T.reshape(1, self.pred_len)

                np.fill_diagonal(self.forecast_tensor[j:, :, i], forecasts)

        self.total_params = len(arima_model.params.index[arima_model.params.index != 'sigma2'])

        display_model_info(self)