"""
__author__ = "Mohammed Aldosari"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

from tqdm.notebook import tqdm
from modeling.statmodeling.model import Model
from util.tools import display_model_info
from numpy import ndarray
import numpy as np

class RandomWalk(Model):
    def __init__(self, args):
        self.model_name = 'Random Walk'
        super().__init__(args)

        if self.skip_insample is None:
            None
        elif self.skip_insample < 0 or self.skip_insample == 0 or self.skip_insample >= len(self.data):
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
                np.fill_diagonal(self.forecast_tensor[j:, :, i], self.data.iloc[self.train_size - 1 + j, i])
        self.total_params = 0
        if self.args.get('internal_diagnose') is None:
            display_model_info(self)


