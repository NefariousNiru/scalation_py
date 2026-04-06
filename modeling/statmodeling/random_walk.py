"""
__author__ = "Mohammed Aldosari", "Nirupom Bose Roy"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

from tqdm.auto import tqdm
from modeling.statmodeling.model import Model
from util.tools import display_model_info
from numpy import ndarray
import numpy as np
from statistics import NormalDist


class RandomWalk(Model):
    def __init__(self, args):
        self.model_name = "Random Walk"
        super().__init__(args)

        if self.skip_insample is None:
            pass
        elif (
            self.skip_insample < 0
            or self.skip_insample == 0
            or self.skip_insample >= len(self.data)
        ):
            raise ValueError(
                f"Invalid value for 'skip_insample'. Expected one of the following:\n"
                f"-  1 or a positive integer less than {len(self.data)}.\n"
                f"- None, to indicate out-of-sample validation.\n"
                f"Received: {self.skip_insample}."
            )

    def _get_training_series_for_feature(self, feature_idx: int) -> np.ndarray:
        """
        Return the series used to estimate RW residual variance for one feature.
        """
        if self.skip_insample is None:
            series = self.data.iloc[: self.train_size, feature_idx].values
        else:
            series = self.data.iloc[: self.train_size, feature_idx].values

        return np.asarray(series, dtype=float)

    def _estimate_rw_sigma(self, feature_idx: int) -> float:
        """
        Estimate one-step random walk innovation std from first differences
        on the training series.
        """
        series = self._get_training_series_for_feature(feature_idx)

        if len(series) < 2:
            return 0.0

        diffs = np.diff(series)
        diffs = diffs[~np.isnan(diffs)]

        if len(diffs) == 0:
            return 0.0

        sigma = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float(np.std(diffs))
        if np.isnan(sigma):
            sigma = 0.0

        return sigma

    def train_test(self) -> None:
        self.forecast_tensor: ndarray[float] = np.full(
            shape=(self.test_size, self.pred_len, self.n_features),
            fill_value=np.nan,
        )

        if self.forecast_type == "interval":
            self.lower_forecast_tensor: ndarray[float] = np.full(
                shape=(self.test_size, self.pred_len, self.n_features),
                fill_value=np.nan,
            )
            self.upper_forecast_tensor: ndarray[float] = np.full(
                shape=(self.test_size, self.pred_len, self.n_features),
                fill_value=np.nan,
            )

            nominal_coverage = 1.0 - self.interval_alpha
            z_value = NormalDist().inv_cdf(1.0 - self.interval_alpha / 2.0)

        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0

        for i in tqdm(range(self.n_features)):
            sigma_i = None
            if self.forecast_type == "interval":
                sigma_i = self._estimate_rw_sigma(i)

            for j in tqdm(range(self.test_size - sample_offset)):
                last_observed = self.data.iloc[self.train_size - 1 + j, i]

                # Point forecast
                np.fill_diagonal(
                    self.forecast_tensor[j:, :, i],
                    last_observed,
                )

                # Interval forecast
                if self.forecast_type == "interval":
                    for h in range(self.pred_len):
                        row_idx = j + h
                        if row_idx >= self.test_size:
                            break

                        horizon = h + 1
                        interval_half_width = z_value * np.sqrt(horizon) * sigma_i

                        lower_h = last_observed - interval_half_width
                        upper_h = last_observed + interval_half_width

                        self.lower_forecast_tensor[row_idx, h, i] = lower_h
                        self.upper_forecast_tensor[row_idx, h, i] = upper_h

        self.total_params = 0

        if self.args.get("internal_diagnose") is None:
            display_model_info(self)
