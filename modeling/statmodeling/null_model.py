"""
__author__ = "Mohammed Aldosari"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import time
from tqdm.notebook import tqdm
from modeling.statmodeling.model import Model
from util.tools import display_model_info
from numpy import ndarray
import numpy as np

class NullModel(Model):
    def __init__(self, args):
        self.model_name = 'Null Model'
        super().__init__(args)
        if self.skip_insample is None:
            None
        elif self.skip_insample < 0 or self.skip_insample >= len(self.data):
            raise ValueError(
                f"Invalid value for 'skip_insample'. Expected one of the following:\n"
                f"-  1 or a positive integer less than {len(self.data)}.\n"
                f"- None, to indicate out-of-sample validation.\n"
                f"Received: {self.skip_insample}."
            )

    def train_test(self) -> None:
        self.forecast_tensor: ndarray[float] = np.full(shape=(self.test_size, self.pred_len, self.n_features),
                                                    fill_value=np.nan)

        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0

        for i in tqdm(range(self.n_features)):
            for j in tqdm(range(self.test_size - sample_offset)):
                if self.skip_insample is None:
                    #self.sample_mean = self.data.iloc[0:self.train_size, i].mean()
                    np.fill_diagonal(self.forecast_tensor[j:, :, i], self.sample_mean_normalized.iloc[:,i])
                else:
                    """if self.debugging:
                        self.sample_mean = self.data.iloc[self.skip_insample:self.original_shape[0], i].mean()
                    else:
                        self.sample_mean = self.data.iloc[self.skip_insample:, i].mean()"""
                    np.fill_diagonal(self.forecast_tensor[j:, :, i], self.sample_mean_normalized.iloc[:,i])

            if self.debugging:
                print(f'sample mean: {self.sample_mean}')

        self.total_params = 0

        display_model_info(self)