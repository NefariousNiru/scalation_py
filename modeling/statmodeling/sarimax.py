"""
__author__ = "Nirupom Bose Roy"
__date__ = "2026-04-04"
__version__ = "2.0"
__license__ = "MIT style license file"
"""

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_
from modeling.statmodeling.model import Model
from util.tools import display_model_info


class SARIMAX(Model):
    """
    Target-only SARIMAX baseline for ILI-style weekly series.

    What changed in this version
    ----------------------------
    1. Keeps the original trainNtest() behavior for your baseline pipeline.
    2. Adds build_origin_forecast_dataset(), which generates aligned rolling-origin
       forecast matrices over any origin range.
    3. Exposes the exact artifacts needed for a publishable hybrid pipeline:
       - history cutoff for each origin
       - SARIMAX H-step forecast matrix
       - true H-step matrix
       - residual H-step matrix

    Notes
    -----
    - This class still forecasts ONLY the target.
    - Hybrid learning happens outside this class.
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
        self.trend = args["trend"]
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

        self.exog_df = self._build_exog_dataframe()

    # ============================================================
    # Public baseline API
    # ============================================================

    def train_test(self) -> None:
        """
        Standard outer test rolling-origin evaluation used by the existing
        statistical-model pipeline.

        Produces:
        - self.forecast_tensor
        - self.forecast_tensor_original (via Model.trainNtest())
        - self.rolling_origin_forecasts
        - self.rolling_origin_truths
        - self.rolling_origin_train_end_idx
        - self.one_step_pred_series
        """
        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0

        origin_start_idx = self.train_size
        origin_end_idx_exclusive = len(self.data) - self.pred_len + 1

        artifacts = self.build_origin_forecast_dataset(
            origin_start_idx=origin_start_idx,
            origin_end_idx_exclusive=origin_end_idx_exclusive,
            return_original_scale=False,
            verbose=True,
        )

        self.rolling_origin_forecasts = artifacts["forecast_matrix_model"]
        self.rolling_origin_truths = artifacts["truth_matrix_model"]
        self.rolling_origin_train_end_idx = artifacts["train_end_idx"]

        n_origins = self.rolling_origin_forecasts.shape[0]

        expected_n_origins = self.test_size - sample_offset
        if n_origins != expected_n_origins:
            raise ValueError(
                f"Outer test origin count mismatch. "
                f"Expected {expected_n_origins}, got {n_origins}."
            )

        self.forecast_tensor: ndarray[float] = np.full(
            shape=(self.test_size, self.pred_len, 1),
            fill_value=np.nan,
        )

        self.one_step_pred_series: ndarray[float] = np.full(
            shape=(len(self.data),),
            fill_value=np.nan,
        )

        for j in range(n_origins):
            start = self.rolling_origin_train_end_idx[j]
            self.one_step_pred_series[start] = self.rolling_origin_forecasts[j, 0]

            np.fill_diagonal(
                self.forecast_tensor[j:, :, 0],
                self.rolling_origin_forecasts[j].reshape(1, self.pred_len),
            )

        self.total_params = artifacts["last_result_params_count"]
        display_model_info(self)

    def build_origin_forecast_dataset(
        self,
        origin_start_idx: int,
        origin_end_idx_exclusive: int,
        return_original_scale: bool = True,
        verbose: bool = False,
    ) -> dict:
        """
        Build rolling-origin H-step forecast artifacts over an arbitrary origin range.

        Parameters
        ----------
        origin_start_idx:
            First origin index. Training uses rows [0 : origin_idx),
            forecasting starts at row origin_idx.

        origin_end_idx_exclusive:
            One past the last origin index.

        return_original_scale:
            If True, includes original-scale matrices.

        verbose:
            If True, shows a tqdm progress bar.

        Returns
        -------
        dict with:
            train_end_idx
            forecast_matrix_model
            truth_matrix_model
            residual_matrix_model
            forecast_matrix_original
            truth_matrix_original
            residual_matrix_original
            origin_dates
            last_result_params_count
        """
        self._validate_origin_range(
            origin_start_idx=origin_start_idx,
            origin_end_idx_exclusive=origin_end_idx_exclusive,
        )

        y_all = self.data[self.target.lower()].reset_index(drop=True).astype(float)
        X_all = self.exog_df.reset_index(drop=True).astype(float)

        n_origins = origin_end_idx_exclusive - origin_start_idx

        forecast_matrix_model = np.full((n_origins, self.pred_len), np.nan, dtype=float)
        truth_matrix_model = np.full((n_origins, self.pred_len), np.nan, dtype=float)
        train_end_idx = np.full((n_origins,), -1, dtype=int)

        sarimax_result = None
        last_result_params_count = 0

        iterator = range(n_origins)
        if verbose:
            iterator = tqdm(iterator, desc="Rolling forecast")

        for row in iterator:
            origin_idx = origin_start_idx + row
            train_end_idx[row] = origin_idx

            y_train = y_all.iloc[:origin_idx]
            X_train = X_all.iloc[:origin_idx, :]

            if sarimax_result is None or row % self.rc == 0:
                sarimax_result = self._fit_model(endog=y_train, exog=X_train)
                last_result_params_count = len(
                    sarimax_result.params.index[sarimax_result.params.index != "sigma2"]
                )
            else:
                new_y = y_all.iloc[origin_idx - 1 : origin_idx]
                new_X = X_all.iloc[origin_idx - 1 : origin_idx, :]
                sarimax_result = sarimax_result.append(
                    endog=new_y,
                    exog=new_X,
                    refit=False,
                )

            start = origin_idx
            end = origin_idx + self.pred_len - 1

            X_future = X_all.iloc[start : end + 1, :]
            y_future = y_all.iloc[start : end + 1].to_numpy(dtype=float)

            forecasts = sarimax_result.predict(
                start=start,
                end=end,
                exog=X_future,
                dynamic=True,
            ).values.reshape(self.pred_len)

            forecast_matrix_model[row, :] = forecasts
            truth_matrix_model[row, :] = y_future

        residual_matrix_model = truth_matrix_model - forecast_matrix_model

        result = {
            "train_end_idx": train_end_idx,
            "forecast_matrix_model": forecast_matrix_model,
            "truth_matrix_model": truth_matrix_model,
            "residual_matrix_model": residual_matrix_model,
            "origin_dates": self._get_origin_dates(
                origin_start_idx=origin_start_idx,
                origin_end_idx_exclusive=origin_end_idx_exclusive,
            ),
            "last_result_params_count": last_result_params_count,
        }

        if return_original_scale:
            result["forecast_matrix_original"] = self._to_original_scale(
                forecast_matrix_model
            )
            result["truth_matrix_original"] = self._to_original_scale(
                truth_matrix_model
            )
            result["residual_matrix_original"] = (
                result["truth_matrix_original"] - result["forecast_matrix_original"]
            )

        return result

    # ============================================================
    # Internal helpers
    # ============================================================

    def _validate_origin_range(
        self,
        origin_start_idx: int,
        origin_end_idx_exclusive: int,
    ) -> None:
        n_total = len(self.data)

        if origin_start_idx <= 0:
            raise ValueError(f"origin_start_idx must be > 0. Got {origin_start_idx}.")

        if origin_end_idx_exclusive <= origin_start_idx:
            raise ValueError(
                f"origin_end_idx_exclusive must be > origin_start_idx. "
                f"Got start={origin_start_idx}, end={origin_end_idx_exclusive}."
            )

        if origin_end_idx_exclusive > (n_total - self.pred_len + 1):
            raise ValueError(
                f"Origin range exceeds valid forecasting boundary. "
                f"Max valid end_exclusive is {n_total - self.pred_len + 1}, "
                f"got {origin_end_idx_exclusive}."
            )

    def _get_origin_dates(
        self,
        origin_start_idx: int,
        origin_end_idx_exclusive: int,
    ) -> pd.Series:
        if hasattr(self, "dates") and self.dates is not None:
            dates = pd.to_datetime(pd.Series(self.dates)).reset_index(drop=True)
            return dates.iloc[origin_start_idx:origin_end_idx_exclusive].reset_index(
                drop=True
            )
        raise ValueError("Aligned dates are unavailable on the SARIMAX object.")

    def _to_original_scale(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert from model space back to original scale.

        Supports:
        - normalization=None
        - normalization='log'
        """
        if self.normalization is None:
            return arr.copy()

        if self.normalization == "log":
            return np.expm1(arr)

        raise NotImplementedError(
            f"_to_original_scale is not implemented for normalization={self.normalization!r}."
        )

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
        d = np.minimum(d, period - d)
        z = 1.0 + (d**2) / (nu * (s**2))
        return z ** (-(nu + 1.0) / 2.0)

    def _build_exog_dataframe(self) -> pd.DataFrame:
        """
        Create time-safe engineered exogenous regressors aligned directly to the
        already-loaded target dataframe length.
        """
        T = len(self.data)

        if hasattr(self, "dates") and self.dates is not None:
            dt = pd.to_datetime(pd.Series(self.dates)).reset_index(drop=True)

            if len(dt) != T:
                raise ValueError(
                    f"Length mismatch: len(self.dates)={len(dt)} but len(self.data)={T}."
                )

        elif isinstance(self.data.index, pd.DatetimeIndex):
            dt = pd.to_datetime(self.data.index)

        else:
            raise ValueError(
                "Could not obtain aligned dates for exogenous construction. "
                "Expected self.dates from load_data() or a DatetimeIndex on self.data."
            )

        self._exog_generation_dates = pd.DatetimeIndex(dt)

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

        exog_df = pd.DataFrame(exog)

        if len(exog_df) != T:
            raise ValueError(
                f"Constructed exog_df has {len(exog_df)} rows but expected {T}."
            )

        return exog_df

    def _fit_model(self, endog: pd.Series, exog: pd.DataFrame):
        """
        Fit one SARIMAX model instance.
        """
        if isinstance(endog, pd.DataFrame):
            if endog.shape[1] != 1:
                raise ValueError(f"Expected univariate endog, got shape {endog.shape}")
            endog = endog.iloc[:, 0]

        endog = pd.Series(endog).astype(float).reset_index(drop=True)
        exog = pd.DataFrame(exog).astype(float).reset_index(drop=True)

        model = SARIMAX_(
            endog=endog,
            exog=exog,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.P, self.D, self.Q, self.s),
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        result = model.fit(
            method=self.fit_method,
            maxiter=self.optimizer_maxiter,
            disp=False,
        )
        return result

    def audit_exog_alignment(self) -> None:
        if len(self.exog_df) != len(self.data):
            raise ValueError(
                f"Length mismatch: len(exog_df)={len(self.exog_df)} vs len(data)={len(self.data)}"
            )

        if self.exog_df.isna().any().any():
            bad_cols = self.exog_df.columns[self.exog_df.isna().any()].tolist()
            raise ValueError(f"Exog contains NaNs in columns: {bad_cols}")

        debug_df = pd.DataFrame()
        debug_df["target"] = self.data[self.target.lower()].values
        for c in self.exog_df.columns[: min(5, len(self.exog_df.columns))]:
            debug_df[c] = self.exog_df[c].values

        print("\n[Alignment audit] head:")
        print(debug_df.head(10))
        print("\n[Alignment audit] tail:")
        print(debug_df.tail(10))

    def audit_exog_generation_dates(self) -> None:
        if hasattr(self, "dates") and self.dates is not None:
            target_dates = pd.to_datetime(pd.Series(self.dates)).reset_index(drop=True)
            gen_dates = pd.to_datetime(
                pd.Series(self._exog_generation_dates)
            ).reset_index(drop=True)

            if len(target_dates) != len(gen_dates):
                raise ValueError(
                    f"Date length mismatch: target_dates={len(target_dates)} vs gen_dates={len(gen_dates)}"
                )

            if not target_dates.equals(gen_dates):
                cmp = pd.DataFrame(
                    {
                        "target_date": target_dates,
                        "exog_generation_date": gen_dates,
                    }
                )
                print(cmp.head(15))
                print(cmp.tail(15))
                raise ValueError(
                    "Exog generation dates do not match load_data() dates row-by-row"
                )
        else:
            raise ValueError(
                "self.dates is unavailable. Cannot prove row-by-row date alignment."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        if "args" in state and state["args"] is not None:
            state["args"] = dict(state["args"])
        return state
