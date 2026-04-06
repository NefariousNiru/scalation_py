"""
__author__ = "Mohammed Aldosari", "Nirupom Bose Roy"
__date__ = 11/04/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
import statsmodels.api as sm


def plot_time_series(self) -> None:
    """
    Plot the target time series for initial visualization.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    data = self.data_
    plt.subplots(figsize=(7, 4))
    plt.plot(
        data[[self.target.lower()]],
        color="black",
        marker="o",
        linewidth=0.5,
        markersize=1,
    )
    plt.title(self.dataset + " " + self.target)
    plt.grid(False)
    file_path = os.path.join(self.folder_path_plots, str(self.dataset) + ".png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.ylabel("Original Scale")
    plt.xlabel("Time")
    plt.show()
    plt.close()


def plot_acf(self) -> None:
    """
    Plot the autocorrelation function (ACF).

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    data = self.data_[[self.target.lower()]]
    data = data.iloc[self.start_index_acf_pacf :]
    if self.diff_order_acf_pacf is not None:
        for _ in range(self.diff_order_acf_pacf):
            data = data.diff().dropna()
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=self.pred_len)
    file_path = os.path.join(self.folder_path_plots, str(self.dataset) + "_ACF.png")
    plt.title("ACF for " + self.target.lower())
    plt.savefig(file_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_pacf(self) -> None:
    """
    Plot the partial autocorrelation function (PACF).

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    data = self.data_[[self.target.lower()]]
    data = data.iloc[self.start_index_acf_pacf :]
    if self.diff_order_acf_pacf is not None:
        for _ in range(self.diff_order_acf_pacf):
            data = data.diff().dropna()
    sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=self.pred_len)
    file_path = os.path.join(self.folder_path_plots, str(self.dataset) + "_PACF.png")
    plt.title("PACF for " + self.target.lower())
    plt.savefig(file_path, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_forecasts(self) -> None:
    """
    Plot observed data together with point forecasts or prediction intervals
    for each requested horizon.
    """

    legend_x_axis = False
    show_title = False
    show_grid = True
    legend_frameon = True

    for h in self.horizons:
        # ---------------------------------------------------------------------
        # Select scale-specific tensors
        # ---------------------------------------------------------------------
        if "normalized" in self.plot_scope_scale:
            forecasts_all = self.forecast_tensor
            lower_all = (
                self.lower_forecast_tensor if self.forecast_type == "interval" else None
            )
            upper_all = (
                self.upper_forecast_tensor if self.forecast_type == "interval" else None
            )

            if "test" in self.plot_scope_scale:
                actual_df = self.data.iloc[self.train_size :, :]
            else:
                actual_df = self.data

            idx = actual_df.index

        elif "original" in self.plot_scope_scale:
            forecasts_all = self.forecast_tensor_original
            lower_all = (
                self.lower_forecast_tensor_original
                if self.forecast_type == "interval"
                else None
            )
            upper_all = (
                self.upper_forecast_tensor_original
                if self.forecast_type == "interval"
                else None
            )

            if "test" in self.plot_scope_scale:
                actual_df = self.data_.iloc[self.train_size :, :]
            else:
                actual_df = self.data_

            idx = actual_df.index

        else:
            raise ValueError(
                "Invalid plot_scope_scale. Expected 'all_normalized', "
                "'all_original', 'test_normalized', or 'test_original'."
            )

        # ---------------------------------------------------------------------
        # Select target series for the requested horizon
        # ---------------------------------------------------------------------
        if self.features == "ms":
            yp = forecasts_all[:, h - 1, self.target_feature].reshape(-1)
            actual = actual_df.iloc[:, self.target_feature].values.reshape(-1)

            if self.forecast_type == "interval":
                lower = lower_all[:, h - 1, self.target_feature].reshape(-1)
                upper = upper_all[:, h - 1, self.target_feature].reshape(-1)

        elif self.features == "m":
            yp = forecasts_all[:, h - 1, self.target_feature].reshape(-1)
            actual = actual_df.iloc[:, self.target_feature].values.reshape(-1)

            if self.forecast_type == "interval":
                lower = lower_all[:, h - 1, self.target_feature].reshape(-1)
                upper = upper_all[:, h - 1, self.target_feature].reshape(-1)

        elif self.features == "s":
            yp = forecasts_all[:, h - 1, -1].reshape(-1)
            actual = actual_df.iloc[:, self.target_feature].values.reshape(-1)

            if self.forecast_type == "interval":
                lower = lower_all[:, h - 1, -1].reshape(-1)
                upper = upper_all[:, h - 1, -1].reshape(-1)

        else:
            raise ValueError(
                "Invalid features setting. Expected one of 'ms', 'm', or 's'."
            )

        forecast_idx = idx[-yp.shape[0] :]

        # ---------------------------------------------------------------------
        # Shared labels
        # ---------------------------------------------------------------------
        if self.normalization is None or "original" in self.plot_scope_scale:
            ylabel = "Original Scale"
        elif self.normalization == "z-score" and "normalized" in self.plot_scope_scale:
            ylabel = "Normalized Scale"
        elif (
            self.normalization == "log" or self.normalization == "log_z-score"
        ) and "normalized" in self.plot_scope_scale:
            ylabel = "Transformed Scale"
        else:
            ylabel = "Value"

        # ---------------------------------------------------------------------
        # Point forecasts
        # ---------------------------------------------------------------------
        if self.forecast_type == "point":
            plt.subplots(figsize=(4, 2.5))

            plt.plot(
                idx,
                actual,
                color="#ef4470",
                linewidth=0.5,
                marker="o",
                markersize=1,
                label="y",
            )
            plt.plot(
                forecast_idx,
                yp,
                color="#26abe3",
                marker="o",
                markersize=1,
                linewidth=0.5,
                label=f"yp h = {h}",
            )

            plt.ylabel(ylabel)
            plt.xlabel("Time")

            if show_title:
                plt.title(
                    "Model: "
                    + self.model_name
                    + " Validation: "
                    + self.validation
                    + "\nDataset: "
                    + self.dataset
                    + " Target: "
                    + self.target
                    + " Forecast type: "
                    + self.forecast_type.capitalize()
                )

            if show_grid:
                plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
            else:
                plt.grid(False)

            if legend_x_axis:
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=2,
                    frameon=legend_frameon,
                )
            else:
                plt.legend(frameon=legend_frameon)

            file_path = os.path.join(
                self.folder_path_plots,
                str(self.validation)
                + "_"
                + str(self.model_name)
                + "_"
                + str(self.pred_len)
                + "_"
                + str(h + 1)
                + "_"
                + self.args["plot_scope_scale"]
                + "_"
                + self.args["target"]
                + ".pdf",
            )

            plt.savefig(file_path, bbox_inches="tight")
            plt.ticklabel_format(style="plain", axis="y")
            plt.show()
            plt.close()

        # ---------------------------------------------------------------------
        # Interval forecasts
        # ---------------------------------------------------------------------
        elif self.forecast_type == "interval":
            plt.subplots(figsize=(4, 2.5))

            plt.plot(
                idx,
                actual,
                color="#ef4470",
                linewidth=0.5,
                marker="o",
                markersize=1,
                label="y",
            )

            plt.plot(
                forecast_idx,
                yp,
                color="#26abe3",
                marker="o",
                markersize=1,
                linewidth=0.5,
                label=f"yp h = {h}",
            )

            coverage_pct = int(round((1.0 - self.interval_alpha) * 100))

            plt.fill_between(
                forecast_idx,
                lower,
                upper,
                alpha=0.2,
                color="#26abe3",
                label=f"{coverage_pct}% PI",
            )

            plt.ylabel(ylabel)
            plt.xlabel("Time")

            if show_title:
                plt.title(
                    "Model: "
                    + self.model_name
                    + " Validation: "
                    + self.validation
                    + "\nDataset: "
                    + self.dataset
                    + " Target: "
                    + self.target
                    + " Forecast type: "
                    + self.forecast_type.capitalize()
                )

            if show_grid:
                plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
            else:
                plt.grid(False)

            if legend_x_axis:
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=3,
                    frameon=legend_frameon,
                )
            else:
                plt.legend(frameon=legend_frameon)

            file_path = os.path.join(
                self.folder_path_plots,
                str(self.validation)
                + "_"
                + str(self.model_name)
                + "_"
                + str(self.pred_len)
                + "_"
                + str(h + 1)
                + "_"
                + self.args["plot_scope_scale"]
                + "_"
                + self.args["target"]
                + "_interval.pdf",
            )

            plt.savefig(file_path, bbox_inches="tight")
            plt.ticklabel_format(style="plain", axis="y")
            plt.show()
            plt.close()

        else:
            raise ValueError(
                f"Invalid forecast_type: {self.forecast_type}. "
                f"Expected 'point' or 'interval'."
            )
