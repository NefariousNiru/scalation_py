"""
__author__ = "Mohammed Aldosari"
__date__ = "11/03/24"
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from modeling.statmodeling.model import Model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import yule_walker
from numpy import ndarray
from util.tools import display_model_info

class AR(Model):

    def __init__(self, args):
        self.p = args['p']
        self.rc = args['rc']
        self.trend = args['trend']
        self.model_name = f'AR({self.p}) rc = {self.rc}'
        super().__init__(args)

        if self.skip_insample is not None and self.skip_insample < self.p:
            raise ValueError(
                f"Invalid value for 'skip_insample'. Expected to be at least {self.p}.\n"
                f"Received: {self.skip_insample}."
            )

    def train_test(self) -> None:
        self.forecast_tensor: ndarray = np.full(
            shape=(self.test_size, self.pred_len, self.n_features),
            fill_value=np.nan
        )

        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0

        for i in tqdm(range(self.n_features)):
            for j in tqdm(range(self.test_size - sample_offset)):
                if j % self.rc == 0:
                    if self.skip_insample is None:
                        train_data = self.data.iloc[0:self.train_size + j, i]
                    else:
                        train_data = self.data.iloc[:, i]  # use entire data for in-sample.
                    
                    if self.args['ar_type'] == 'OLS':
                        ar_model = AutoReg(
                            train_data,
                            lags=self.p,
                            trend=self.trend,
                            seasonal=False
                        ).fit()
                    elif self.args['ar_type'] == 'yule_walker_adjusted':
                        rho, sigma = yule_walker(train_data, order=self.p, method='adjusted')
                        intercept = np.mean(train_data) * (1 - np.sum(rho)) if self.trend == 'c' else 0
                        ar_model = (rho, sigma, intercept)
                    elif self.args['ar_type'] == 'yule_walker_mle':
                        rho, sigma = yule_walker(train_data, order=self.p, method='mle')
                        intercept = np.mean(train_data) * (1 - np.sum(rho)) if self.trend == 'c' else 0
                        ar_model = (rho, sigma, intercept)
                else:
                    if self.skip_insample is None:
                        new_data = self.data.iloc[self.train_size + j - 1:self.train_size + j, i]
                        train_data = pd.concat([train_data, new_data])

                        if self.args['ar_type'] == 'OLS':
                            ar_model = ar_model.append(endog=new_data, refit=False)
                        elif self.args['ar_type'] in ['yule_walker_adjusted', 'yule_walker_mle']:
                            rho, sigma = yule_walker(train_data, order=self.p, method=self.args['ar_type'].split('_')[2])
                            intercept = np.mean(train_data) * (1 - np.sum(rho)) if self.trend == 'c' else 0
                            ar_model = (rho, sigma, intercept)
                            

                start = self.train_size + j
                end = start + self.pred_len - 1

                if self.args['ar_type'] == 'OLS':
                    forecasts = ar_model.predict(
                        start=start, end=end, dynamic=True
                    ).values.T.reshape(1, self.pred_len)
                elif self.args['ar_type'] in ['yule_walker_adjusted', 'yule_walker_mle']:
                    params, _, intercept = ar_model
                    params = np.concatenate(([intercept], params))
                    model = AutoReg(
                        train_data,
                        lags=self.p,
                        trend=self.trend
                    )
                    forecasts = model.predict(
                        params = params,
                        start = start, end=end, dynamic = True
                    ).values.T.reshape(1, self.pred_len)

                np.fill_diagonal(self.forecast_tensor[j:, :, i], forecasts)

        if isinstance(ar_model, tuple):
            self.total_params =  len(ar_model[0]) + 1 if intercept else len(ar_model[0])
            print(f'params: {params}')
        else:
            self.total_params = len(ar_model.params)
            print(f"params: {ar_model.params.values}")
        display_model_info(self)
