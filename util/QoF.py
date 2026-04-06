"""
__author__ = "Mohammed Aldosari", "Nirupom Bose Roy"
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

pd.set_option("display.float_format", lambda x: "%.3f" % x)

"""
QoF:
    A module for evaluating the quality/goodness of fit.
    Supports point-forecast metrics and interval-forecast metrics.
"""


# ============================================================================
# Point forecast metrics
# ============================================================================


def smape(self, y: np.ndarray, yp: np.ndarray) -> float:
    epsilon = 1e-8
    numerator = np.abs(y - yp)
    denominator = np.abs(y) + np.abs(yp) + epsilon
    smapes = np.mean(200 * (numerator / denominator), axis=0)
    smapes = smapes.flatten()
    return smapes


def mae(self, y: np.ndarray, yp: np.ndarray) -> float:
    maes = np.abs(y - yp)
    maes = np.mean(maes, axis=0)
    if self.args.get("internal_diagnose"):
        return maes
    else:
        maes = maes.flatten()
        return maes


def sst(self, y: np.ndarray, sample_mean) -> float:
    ssts = (y - np.squeeze(sample_mean.values)) ** 2
    ssts = np.sum(ssts, axis=0)
    return ssts


def sse(self, y: np.ndarray, yp: np.ndarray) -> float:
    sses = (y - yp) ** 2
    sses = np.sum(sses, axis=0)
    return sses


def r2q(self, y: np.ndarray, yp: np.ndarray, sample_mean) -> float:
    sse_ = sse(self, y, yp)
    sst_ = sst(self, y, sample_mean)

    r2 = 1 - (sse_ / sst_)
    r2 = r2.flatten()
    return r2


def mse(self, y: np.ndarray, yp: np.ndarray) -> float:
    mses = (y - yp) ** 2
    mses = np.mean(mses, axis=0)
    mses = mses.flatten()
    return mses


def rmse(self, y: np.ndarray, yp: np.ndarray) -> float:
    rmses = np.sqrt(mse(self, y, yp))
    rmses = rmses.flatten()
    return rmses


def corr(self, y: np.ndarray, yp: np.ndarray) -> float:
    mean_y = np.mean(y, axis=0)
    mean_yp = np.mean(yp, axis=0)

    covariance = np.mean((y - mean_y) * (yp - mean_yp), axis=0)

    std_y = np.std(y, axis=0)
    std_yp = np.std(yp, axis=0)
    correlations = covariance / (std_y * std_yp + 1e-8)

    return correlations


def bias(self, y: np.ndarray, yp: np.ndarray) -> float:
    biases = np.mean(yp - y, axis=0)
    return biases


def mase(self, y: np.ndarray, yp: np.ndarray, maes_naive=None) -> float:
    maes = np.abs(y - yp)
    maes = np.mean(maes, axis=0)
    maes = maes.flatten()
    if self.args.get("internal_diagnose"):
        mases = maes / maes
    else:
        mases = maes / maes_naive

    return mases


def mape(self, y: np.ndarray, yp: np.ndarray) -> float:
    epsilon = 1e-8
    mape_values = np.mean(100 * np.abs((y - yp) / (y + epsilon)), axis=0)
    return mape_values


# ============================================================================
# Interval forecast metrics
# ============================================================================


def picp(self, y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Prediction Interval Coverage Probability.
    Fraction of observations falling inside the prediction interval.
    """
    inside = ((y >= lower) & (y <= upper)).astype(float)
    return np.mean(inside, axis=0).flatten()


def ace(self, y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Average Coverage Error = PICP - PINC
    where PINC = 1 - interval_alpha.
    """
    nominal_coverage = 1.0 - self.interval_alpha
    return (picp(self, y, lower, upper) - nominal_coverage).flatten()


def pinaw(self, y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Prediction Interval Normalized Average Width.
    Normalized by the range of observed values in the evaluation sample.
    """
    widths = upper - lower
    y_range = np.nanmax(y, axis=0) - np.nanmin(y, axis=0)
    y_range = np.where(np.abs(y_range) < 1e-8, 1e-8, y_range)
    pinaw_ = np.mean(widths, axis=0) / y_range
    return pinaw_.flatten()


def mis(self, y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Mean Interval Score for a central (1 - alpha) prediction interval.
    Lower is better.
    """
    alpha = self.interval_alpha

    interval_width = upper - lower
    below_penalty = (2.0 / alpha) * (lower - y) * (y < lower)
    above_penalty = (2.0 / alpha) * (y - upper) * (y > upper)

    score = interval_width + below_penalty + above_penalty
    return np.mean(score, axis=0).flatten()


def wis(
    self, y: np.ndarray, lower: np.ndarray, upper: np.ndarray, yp: np.ndarray
) -> float:
    """
    Weighted Interval Score.

    For the current single-interval implementation, this is set equal to MIS.
    If multiple central intervals / quantiles are added later, generalize this.
    """
    return mis(self, y, lower, upper)


# ============================================================================
# Shared metric collection
# ============================================================================


def get_metrics(
    self,
    actual,
    forecasts,
    sample_mean,
    maes_naive=None,
    lower=None,
    upper=None,
) -> Tuple:
    """
    Compute point metrics, and optionally interval metrics if lower/upper bounds
    are provided.

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    forecasts : np.ndarray
        Point forecasts.
    sample_mean : Any
        Mean of sample used for R^2-like metrics.
    maes_naive : optional
        Naive MAE baseline for MASE.
    lower : np.ndarray, optional
        Lower prediction interval bound.
    upper : np.ndarray, optional
        Upper prediction interval bound.

    Returns
    -------
    Tuple
        Point metrics plus interval metrics (if interval bounds are provided).
    """
    if forecasts.ndim == 1:
        valid_indices = ~np.isnan(forecasts)
        if lower is not None:
            valid_indices &= ~np.isnan(lower)
        if upper is not None:
            valid_indices &= ~np.isnan(upper)
    else:
        valid_indices = ~np.isnan(forecasts).any(axis=1)
        if lower is not None:
            valid_indices &= ~np.isnan(lower).any(axis=1)
        if upper is not None:
            valid_indices &= ~np.isnan(upper).any(axis=1)

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

    n_ = yp.shape[0]

    if lower is not None and upper is not None:
        lo = lower[valid_indices]
        hi = upper[valid_indices]

        picp_ = picp(self, y, lo, hi)
        ace_ = ace(self, y, lo, hi)
        pinaw_ = pinaw(self, y, lo, hi)
        mis_ = mis(self, y, lo, hi)
        wis_ = wis(self, y, lo, hi, yp)
    else:
        picp_ = np.full_like(mse_, np.nan, dtype=float)
        ace_ = np.full_like(mse_, np.nan, dtype=float)
        pinaw_ = np.full_like(mse_, np.nan, dtype=float)
        mis_ = np.full_like(mse_, np.nan, dtype=float)
        wis_ = np.full_like(mse_, np.nan, dtype=float)

    return (
        n_,
        mse_,
        rmse_,
        mae_,
        smape_,
        mase_,
        sse_,
        sst_,
        r2q_,
        corr_,
        bias_,
        picp_,
        ace_,
        pinaw_,
        mis_,
        wis_,
    )


# ============================================================================
# Diagnose / QoF table
# ============================================================================


def diagnose(self):
    if self.args.get("mase_calc") is None:
        from modeling.statmodeling.random_walk import RandomWalk

        args = dotdict()
        args.data_path = self.args.get("data_path")
        args.training_ratio = self.args.get("training_ratio")
        args.normalization = self.args.get("normalization")
        args.dataset = self.args.get("dataset")
        args.target = self.args.get("target")
        args.forecast_type = self.args.get("forecast_type")
        args.interval_alpha = self.args.get("interval_alpha", 0.05)
        args.plot_scope_scale = None
        args.plot_eda = False
        args.qof_calculation_mode = self.args.get("qof_calculation_mode")
        args.horizons = self.horizons
        args.same_n_samples = self.args.get("same_n_samples")
        args.skip_insample = self.args.get("skip_insample")
        args.features = self.args.get("features")
        args.modeling_approach = self.args.get("modeling_approach")
        args.debugging = False
        self.args["mase_calc"] = "done"
        args.mase_calc = "done"
        args.internal_diagonse = True

        rw_model = RandomWalk(args)
        self.mae_normalized_list, self.mae_original_list = rw_model.trainNtest()

    if self.qof is None:
        self.qof = pd.DataFrame(
            columns=[
                "h",
                "n",
                "MSE Normalized",
                "RMSE Normalized",
                "MAE Normalized",
                "sMAPE Normalized",
                "MASE Normalized",
                "SSE Normalized",
                "SST Normalized",
                "R Squared Normalized",
                "Corr Normalized",
                "Bias Normalized",
                "PICP Normalized",
                "ACE Normalized",
                "PINAW Normalized",
                "MIS Normalized",
                "WIS Normalized",
                "MSE Original",
                "RMSE Original",
                "MAE Original",
                "sMAPE Original",
                "MASE Original",
                "SSE Original",
                "SST Original",
                "R Squared Original",
                "Corr Original",
                "Bias Original",
                "PICP Original",
                "ACE Original",
                "PINAW Original",
                "MIS Original",
                "WIS Original",
            ]
        )

    self.qof_metrics = {
        col: np.full(
            (
                self.pred_len,
                self.forecast_tensor.shape[-1] if self.features == "m" else 1,
            ),
            np.nan,
        )
        for col in self.qof.columns
        if col != "h"
    }

    for h in range(self.pred_len):
        if self.features == "m":
            forecast_tensor = self.forecast_tensor[:, h, :]
            forecast_tensor_original = self.forecast_tensor_original[:, h, :]
            actual = self.data.iloc[self.train_size :, :].values
            actual_original = self.data_.iloc[self.train_size :, :].values

            if self.forecast_type == "interval":
                lower_tensor = self.lower_forecast_tensor[:, h, :]
                upper_tensor = self.upper_forecast_tensor[:, h, :]
                lower_tensor_original = self.lower_forecast_tensor_original[:, h, :]
                upper_tensor_original = self.upper_forecast_tensor_original[:, h, :]
            else:
                lower_tensor = upper_tensor = None
                lower_tensor_original = upper_tensor_original = None

        elif self.features == "ms":
            forecast_tensor = self.forecast_tensor[:, h, self.target_feature]
            forecast_tensor_original = self.forecast_tensor_original[
                :, h, self.target_feature
            ]
            actual = self.data.iloc[self.train_size :, self.target_feature].values
            actual_original = self.data_.iloc[
                self.train_size :, self.target_feature
            ].values

            if self.forecast_type == "interval":
                lower_tensor = self.lower_forecast_tensor[:, h, self.target_feature]
                upper_tensor = self.upper_forecast_tensor[:, h, self.target_feature]
                lower_tensor_original = self.lower_forecast_tensor_original[
                    :, h, self.target_feature
                ]
                upper_tensor_original = self.upper_forecast_tensor_original[
                    :, h, self.target_feature
                ]
            else:
                lower_tensor = upper_tensor = None
                lower_tensor_original = upper_tensor_original = None

        elif self.features == "s":
            target_feature = -1
            forecast_tensor = self.forecast_tensor[:, h, target_feature]
            forecast_tensor_original = self.forecast_tensor_original[
                :, h, target_feature
            ]
            actual = self.data.iloc[self.train_size :, target_feature].values
            actual_original = self.data_.iloc[
                self.train_size :, self.target_feature
            ].values

            if self.forecast_type == "interval":
                lower_tensor = self.lower_forecast_tensor[:, h, target_feature]
                upper_tensor = self.upper_forecast_tensor[:, h, target_feature]
                lower_tensor_original = self.lower_forecast_tensor_original[
                    :, h, target_feature
                ]
                upper_tensor_original = self.upper_forecast_tensor_original[
                    :, h, target_feature
                ]
            else:
                lower_tensor = upper_tensor = None
                lower_tensor_original = upper_tensor_original = None

        if self.args.get("internal_diagnose"):
            normalized_metrics = get_metrics(
                self,
                actual,
                forecast_tensor,
                self.sample_mean_normalized,
                lower=lower_tensor,
                upper=upper_tensor,
            )
            original_metrics = get_metrics(
                self,
                actual_original,
                forecast_tensor_original,
                self.sample_mean,
                lower=lower_tensor_original,
                upper=upper_tensor_original,
            )
        else:
            h_for_mase = h
            if len(self.mae_normalized_list) == 1:
                h_for_mase = -1

            normalized_metrics = get_metrics(
                self,
                actual,
                forecast_tensor,
                self.sample_mean_normalized,
                self.mae_normalized_list[h_for_mase],
                lower=lower_tensor,
                upper=upper_tensor,
            )
            original_metrics = get_metrics(
                self,
                actual_original,
                forecast_tensor_original,
                self.sample_mean,
                self.mae_original_list[h_for_mase],
                lower=lower_tensor_original,
                upper=upper_tensor_original,
            )

        all_metrics = normalized_metrics + original_metrics[1:]
        qof_keys = [col for col in self.qof.columns if col != "h"]

        for key, val in zip(qof_keys, all_metrics):
            self.qof_metrics[key][h] = val

    for h in self.horizons:
        row = {"h": h}

        if self.qof_calculation_mode == "single_horizon":
            for col in self.qof.columns:
                if col == "h":
                    continue
                data = self.qof_metrics[col][h - 1]
                row[col] = int(np.nansum(data)) if col == "n" else np.nanmean(data)

        elif self.qof_calculation_mode == "aggregated_horizons":
            for col in self.qof.columns:
                if col == "h":
                    continue
                data = self.qof_metrics[col][0:h].flatten()
                row[col] = int(np.nansum(data)) if col == "n" else np.nanmean(data)

        if self.modeling_approach == "joint":
            self.qof = pd.concat([self.qof, pd.DataFrame([row])], ignore_index=True)

    if self.modeling_approach == "individual":
        self.qof = pd.concat([self.qof, pd.DataFrame([row])], ignore_index=True)

    # -------------------------------------------------------------------------
    # Split into separate point and interval tables
    # -------------------------------------------------------------------------
    point_default_cols = [
        "h",
        "n",
        "MSE Normalized",
        "MAE Normalized",
        "MAE Original",
        "sMAPE Original",
    ]

    interval_default_cols = [
        "h",
        "n",
        "PICP Original",
        "ACE Original",
        "PINAW Original",
        "MIS Original",
        "WIS Original",
    ]

    point_cols = self.args.get("qof_to_display", point_default_cols)
    interval_cols = self.args.get("pi_to_display", interval_default_cols)

    self.qof_point = self.qof[point_cols].copy()

    if self.forecast_type == "interval":
        self.qof_interval = self.qof[interval_cols].copy()
    else:
        self.qof_interval = None

    if self.args.get("internal_diagnose"):
        return self.qof_metrics["MAE Normalized"], self.qof_metrics["MAE Original"]
    else:
        # ---------------------------------------------------------------------
        # Plot only point metrics from qof_to_display
        # ---------------------------------------------------------------------
        for metric in point_cols[2:]:
            custom_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#17becf",
            ]

            plt.figure(figsize=(4.5, 3))

            for i in range(self.qof_metrics[metric].shape[-1]):
                plt.plot(
                    np.arange(1, self.qof_metrics[metric].shape[0] + 1),
                    self.qof_metrics[metric][:, i],
                    label=(
                        self.columns[i]
                        if self.features == "m"
                        else self.columns[self.target_feature]
                    ),
                    color=custom_colors[i % len(custom_colors)],
                    linewidth=0.5,
                    marker="o",
                    markersize=1.2,
                )

            plt.xlabel("Horizons", fontsize=9)
            plt.ylabel(metric, fontsize=9)

            plt.legend(
                fontsize=7.5,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.48),
                ncol=3,
                frameon=False,
            )

            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.grid(True, linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            plt.show()
