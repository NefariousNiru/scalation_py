"""
__author__ = "Nirupom Bose Roy"
__date__ = "2026-04-03"
__version__ = "1.0"
__license__ = "MIT style license file"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numpy import ndarray
from tqdm.auto import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_

from modeling.statmodeling.model import Model
from util.tools import display_model_info


class SARIMAX(Model):
    """
    Target-only SARIMAX baseline for ILI-style weekly series.

    Design goals
    ------------
    - Compatible with the existing Model superclass
    - Forecasts ONLY the target series
    - Uses time-safe engineered exogenous regressors by default
    - Avoids using obviously leakage-prone columns such as % weighted ILI

    Expected args additions
    -----------------------
    args["date_col"]                : str, default "date"
    args["use_fourier_exog"]        : bool, default True
    args["fourier_order"]           : int, default 3
    args["use_bump_exog"]           : bool, default True
    args["bump_mu"]                 : float, default 6.0
    args["bump_s"]                  : float, default 3.0
    args["bump_nu"]                 : float, default 4.0
    args["use_trend_exog"]          : bool, default True
    args["include_provider_exog"]   : bool, default False
    args["include_patient_exog"]    : bool, default False
    args["optimizer_maxiter"]       : int, default 100
    """

    def __init__(self, args):
        self.p = args["p"]
        self.d = args["d"]
        self.q = args["q"]

        self.P = args["P"]
        self.D = args["D"]
        self.Q = args["Q"]
        self.s = args["s"]

        self.rc = args["rc"]
        self.trend = args[
            "trend"
        ]  # usually "n" here, since exog carries mean/trend structure
        self.fit_method = args["fit_method"]

        self.date_col = args.get("date_col", "date")

        self.use_fourier_exog = args.get("use_fourier_exog", True)
        self.fourier_order = args.get("fourier_order", 3)

        self.use_bump_exog = args.get("use_bump_exog", True)
        self.bump_mu = args.get("bump_mu", 6.0)
        self.bump_s = args.get("bump_s", 3.0)
        self.bump_nu = args.get("bump_nu", 4.0)

        self.use_trend_exog = args.get("use_trend_exog", True)

        self.include_provider_exog = args.get("include_provider_exog", False)
        self.include_patient_exog = args.get("include_patient_exog", False)

        self.optimizer_maxiter = args.get("optimizer_maxiter", 100)

        self.model_name = (
            f"SARIMAX({self.p},{self.d},{self.q},{self.P},{self.D},{self.Q},{self.s}) "
            f"rc = {self.rc}"
        )

        super().__init__(args)

        # Build exogenous dataframe aligned to the same clipped/raw series used by the target.
        self.exog_df = self._build_exog_dataframe()

    @staticmethod
    def _student_t_bump(
        week_idx_0_51: np.ndarray,
        mu: float,
        s: float,
        nu: float = 4.0,
    ) -> np.ndarray:
        """
        Smooth circular seasonal bump over epidemiological week.
        """
        period = 52.0
        w = week_idx_0_51.astype(float)
        d = np.abs(w - mu)
        d = np.minimum(d, period - d)  # circular distance
        z = 1.0 + (d**2) / (nu * (s**2))
        return z ** (-(nu + 1.0) / 2.0)

    def _build_exog_dataframe(self) -> pd.DataFrame:
        """
        Create time-safe engineered exogenous regressors aligned directly to the
        already-loaded target dataframe length.

        This avoids trying to reproduce the superclass data loading / clipping logic.
        """
        T = len(self.data)

        # Best case: self.data is already indexed by datetime
        if isinstance(self.data.index, pd.DatetimeIndex):
            dt = pd.to_datetime(self.data.index)
        else:
            # Fallback: reload the CSV only to obtain the date sequence,
            # but do NOT try to reapply clipping logic here.
            raw_df = pd.read_csv(self.data_path).copy()
            raw_df.columns = [str(c).lower() for c in raw_df.columns]

            date_col = self.date_col.lower()
            if date_col not in raw_df.columns:
                raise ValueError(
                    f"date_col='{self.date_col}' not found in dataset columns: {raw_df.columns.tolist()}"
                )

            raw_df[date_col] = pd.to_datetime(raw_df[date_col])
            raw_df = raw_df.sort_values(date_col).reset_index(drop=True)

            if len(raw_df) < T:
                raise ValueError(
                    f"Raw CSV has fewer rows ({len(raw_df)}) than loaded target data ({T})."
                )

            # Align by taking the same number of rows as the loaded data.
            # This assumes load_data preserves temporal order.
            raw_df = raw_df.iloc[-T:].reset_index(drop=True)
            dt = pd.to_datetime(raw_df[date_col])

        iso = pd.Series(dt).dt.isocalendar()
        week_of_year = iso.week.astype(int).to_numpy()
        week_of_year = np.where(week_of_year == 53, 52, week_of_year)
        week_of_year = np.clip(week_of_year, 1, 52) - 1

        t_index = np.arange(T, dtype=float)

        exog = {}
        exog["intercept"] = np.ones(T, dtype=float)

        if self.use_trend_exog:
            exog["time_trend"] = (t_index - t_index.mean()) / (t_index.std() + 1e-8)

        if self.use_fourier_exog:
            for k in range(1, self.fourier_order + 1):
                angle = 2.0 * np.pi * k * week_of_year / 52.0
                exog[f"sin_{k}"] = np.sin(angle)
                exog[f"cos_{k}"] = np.cos(angle)

        if self.use_bump_exog:
            exog["bump"] = self._student_t_bump(
                week_idx_0_51=week_of_year,
                mu=self.bump_mu,
                s=self.bump_s,
                nu=self.bump_nu,
            )

        exog_df = pd.DataFrame(exog, index=self.data.index)

        return exog_df

    def _fit_model(self, endog: pd.Series, exog: pd.DataFrame):
        """
        Fit one SARIMAX model instance.
        """
        model = SARIMAX_(
            endog=endog,
            exog=exog,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self.s),
            trend=None,  # exog already carries intercept/trend structure
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        result = model.fit(
            method=self.fit_method,
            maxiter=self.optimizer_maxiter,
            disp=False,
        )
        return result

    def train_test(self) -> None:
        """
        Rolling-origin evaluation for target-only SARIMAX.

        Produces a forecast tensor of shape:
            (test_size, pred_len, 1)
        """
        if len(self.exog_df) != len(self.data):
            raise ValueError(
                f"Exogenous dataframe length {len(self.exog_df)} does not match target data length {len(self.data)}."
            )

        self.forecast_tensor: ndarray[float] = np.full(
            shape=(self.test_size, self.pred_len, 1),
            fill_value=np.nan,
        )

        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0

        target_col = self.target.lower()
        if target_col not in self.data.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in self.data columns: {self.data.columns.tolist()}"
            )

        y_all = self.data[target_col].reset_index(drop=True)
        X_all = self.exog_df.reset_index(drop=True)

        sarimax_result = None

        for j in tqdm(range(self.test_size - sample_offset), desc="Rolling forecast"):
            train_end_idx = self.train_size + j

            if self.skip_insample is None:
                y_train = y_all.iloc[:train_end_idx]
                X_train = X_all.iloc[:train_end_idx, :]
            else:
                y_train = y_all
                X_train = X_all

            if j % self.rc == 0 or sarimax_result is None:
                sarimax_result = self._fit_model(
                    endog=y_train,
                    exog=X_train,
                )
            else:
                if self.skip_insample is None:
                    new_y = y_all.iloc[train_end_idx - 1 : train_end_idx]
                    new_X = X_all.iloc[train_end_idx - 1 : train_end_idx, :]
                    sarimax_result = sarimax_result.append(
                        endog=new_y,
                        exog=new_X,
                        refit=False,
                    )

            start = self.train_size + j
            end = start + self.pred_len - 1

            X_future = X_all.iloc[start : end + 1, :]

            forecasts = sarimax_result.predict(
                start=start, end=end, exog=X_future, dynamic=True
            ).values.reshape(1, self.pred_len)

            np.fill_diagonal(self.forecast_tensor[j:, :, 0], forecasts)

        if sarimax_result is not None:
            self.total_params = len(
                sarimax_result.params.index[sarimax_result.params.index != "sigma2"]
            )
        else:
            self.total_params = 0

        display_model_info(self)
