"""
__author__ = "Mohammed Aldosari"
__date__ = 11/03/24
__version__ = "1.0"
__license__ = "MIT style license file"
"""

import copy
import os
import time
import numpy as np
from util.data_loading import load_data
from util.data_transforms import transform_data
from util.data_splitting import train_test_split
from util.plotting import plot_time_series, plot_acf, plot_pacf
from util.data_transforms import inverse_transformation
import datetime
from util.plotting import plot_forecasts
from util.QoF import diagnose
from util.tools import display_save_results
np.set_printoptions(suppress=True)

class Model:
    def __init__(self, args):
        self.args = args
        self.modeling_type = 'statistical'
        self.modeling_approach = 'joint'
        self.data_path = args.data_path
        self.skip_insample = self.args.skip_insample
        self.target = self.args.target
        self.dataset = self.args.dataset
        self.start_index_acf_pacf = 0
        self.diff_order_acf_pacf = 0
        self.horizons = self.args.horizons
        self.normalization = self.args.normalization
        self.rc = self.args.rc

        self.same_n_samples = self.args.same_n_samples
        self.plot_eda = self.args.plot_eda
        self.training_ratio = self.args.training_ratio

        self.debugging = self.args.debugging

        self.forecast_type = self.args.forecast_type.lower() if self.args.forecast_type is not None else None
        self.plot_scope_scale = self.args.plot_scope_scale.lower() if self.args.plot_scope_scale is not None else self.args.plot_scope_scale
        self.features = self.args.features.lower() if self.args.features is not None else self.args.features
        self.qof_calculation_mode = self.args.qof_calculation_mode.lower() if self.args.qof_calculation_mode is not None else self.args.qof_calculation_mode
        self.normalization = self.args.normalization.lower() if self.args.normalization is not None else self.args.normalization
        self.modeling_approach = self.args.modeling_approach.lower() if self.args.modeling_approach is not None else self.args.modeling_approach

        if self.training_ratio <= 0 or self.training_ratio >= 100:
            raise ValueError(
                f"Invalid value for 'training_ratio'. Expected a fraction or whole number between 0 and 100. For 80% training ratio, both 0.8 and 80 should work.\n"
                f"Received: {self.training_ratio}."
            )

        self.training_ratio = self.args.training_ratio/100 if isinstance(self.args.training_ratio, int) else self.args.training_ratio


        if self.forecast_type not in ['point', 'interval']:
            raise ValueError(
                f"Invalid value for 'forecast_type'. Expected one of the following 'point' or 'interval'\n"
                f"Received: {self.forecast_type}."
            )

        if self.plot_scope_scale not in ['all_original', 'all_normalized', 'test_original', 'test_normalized', None]:
            raise ValueError(
                f"Invalid value for 'plot_scope_scale'. Expected one of the following 'all_original', 'all_normalized', 'test_original', 'test_normalized', or None\n"
                f"Received: {self.plot_scope_scale}."
            )

        if self.features not in ['ms', 'm', 's']:
            raise ValueError(
                f"Invalid value for 'features'. Expected one of the following 'ms', 'm', 's'\n"
                f"Received: {self.features}."
            )

        if self.qof_calculation_mode not in ['single_horizon', 'aggregated_horizons']:
            raise ValueError(
                f"Invalid value for 'qof_calculation_mode'. Expected one of the following 'single_horizon' or 'aggregated_horizons'\n"
                f"Received: {self.qof_calculation_mode}."
            )

        if type(self.same_n_samples) is not bool:
            raise ValueError(
                f"Invalid value for 'same_n_samples'. Expected a boolean (True or False)\n"
                f"Received: {self.same_n_samples}."
            )

        if type(self.plot_eda) is not bool:
            raise ValueError(
                f"Invalid value for 'plot_eda'. Expected a boolean (True or False)\n"
                f"Received: {self.plot_eda}."
            )

        if type(self.horizons) is not list:
            raise ValueError(
                f"Invalid value for 'horizons'. Expected a list of target horizons.\n"
                f"Received: {self.horizons}."
            )



        if self.normalization not in ['log', 'z-score', 'log_z-score', None]:
            raise ValueError(
                f"Invalid value for 'normalization'. Expected one of the following 'log', 'z-score', or None.\n"
                f"Received: {self.normalization}."
            )

        if self.modeling_approach not in ['joint', 'individual']:
            raise ValueError(
                f"Invalid value for 'modeling_approach'. Expected 'joint' or 'individual'.\n"
                f"Received: {self.modeling_approach}."
            )

        if self.horizons != sorted(self.horizons):
            self.horizons.sort()

        if self.skip_insample is None:
            self.validation = 'Out-of-Sample'
        else:
            self.validation = 'In-Sample'

        self.pred_len = max(self.horizons)

        load_data(self)
        self.columns = self.data.columns
        self.data.columns = self.data.columns.str.lower()
        if self.target.lower() not in self.data.columns.to_list():
            raise ValueError(
                f"Invalid value for 'target'. Expected one of the following: {self.data.columns.to_list()}.\n"
                f"Received: {self.target}."
            )

        if self.features == 's':
            self.data = self.data[[self.target.lower()]]
        """
        if self.debugging:
            new_data = []
            value = self.data.iloc[-1, :].values[0]
            for i in range(12):
                value = value + random.randint(-2,2)
                new_data.append(value)
            print(new_data)
            df_new = pd.DataFrame({'ilitotal': new_data})

            self.data = pd.concat([self.data, df_new], ignore_index=True)"""

        self.target_feature = self.data.columns.get_loc(self.target.lower())
        self.n_features = len(self.data.columns)

        self.qof = None
        self.today = str(datetime.datetime.today().strftime('%Y-%m-%d'))

    def trainNtest(self) -> np.array:

        self.df_raw_len = len(self.data)
        self.folder_path_plots = './plots/' + str(self.validation) + '/' + self.model_name + '/' + str(
            self.dataset) + '/' + str(self.pred_len)
        self.folder_path_results = './results/' + str(self.validation) + '/'

        if not os.path.exists(self.folder_path_plots):
            os.makedirs(self.folder_path_plots)
        if not os.path.exists(self.folder_path_results):
            os.makedirs(self.folder_path_results)

        self.data_ = copy.deepcopy(self.data)




        if self.normalization is not None:
            self.data = transform_data(self)

        if self.skip_insample is None:
            self.train_data, _, self.test_data = train_test_split(self.data, train_ratio=self.training_ratio)
            self.train_size = len(self.train_data)
            self.test_size = len(self.test_data)
        else:
            self.train_size = self.skip_insample
            self.test_size = len(self.data) - self.skip_insample


        if self.skip_insample is None:
            self.sample_mean = self.data_.iloc[:self.train_size,:].mean().to_frame().T
            self.sample_mean_normalized = self.data.iloc[:self.train_size,:].mean().to_frame().T
        else:
            self.sample_mean = self.data_.iloc[self.skip_insample:,:].mean().to_frame().T
            self.sample_mean_normalized = self.data.iloc[self.skip_insample:,:].mean().to_frame().T

        if self.features == 'm':
            self.sample_mean = self.sample_mean
            self.sample_mean_normalized = self.sample_mean_normalized
        elif self.features == 'ms':
            self.sample_mean = self.sample_mean.iloc[:, self.target_feature]
            self.sample_mean_normalized = self.sample_mean_normalized.iloc[:, self.target_feature]
        elif self.features == 's':
            self.sample_mean = self.sample_mean
            self.sample_mean_normalized = self.sample_mean_normalized

        if self.rc is not None:
            if self.skip_insample is not None and self.rc < self.test_size and self.rc is not None:
                raise ValueError(
                    f"Invalid value for 'rc'. Expected a number greater that the test set size ({self.test_size}).\n"
                    f"For in-sample validation, the model will be fitted once on the entire dataset. Retraining cycle is not required.\n"
                    f"Received: {self.rc}."
                )

        if self.plot_eda:
            plot_time_series(self)
            plot_acf(self)
            plot_pacf(self)

        if not self.plot_eda and self.plot_scope_scale is not None:
            plot_time_series(self)

        start_time = time.time()

        if self.modeling_approach == 'joint':
            self.train_test()
            self.forecast_tensor_original = inverse_transformation(self)
            if self.plot_scope_scale is not None:
                plot_forecasts(self)
            if self.args.internal_diagonse:
                self.args.internal_diagnose = True
                mae_normalized_list, mae_original_list = diagnose(self)
            else:
                diagnose(self)

        elif self.modeling_approach == 'individual':
            for h in self.args.horizons:

                self.horizons = [h]
                self.pred_len = h
                self.train_test()
                self.forecast_tensor_original = inverse_transformation(self)
                if self.plot_scope_scale is not None:
                    plot_forecasts(self)
                if self.args.internal_diagonse:
                    self.args.internal_diagnose = True
                    mae_normalized_list, mae_original_list = diagnose(self)
                else:
                    diagnose(self)
                self.args.mase_calc = None
        if self.args.internal_diagonse is None:
            print(self.args)
            display_save_results(self)


        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time:{total_time} seconds. \n")
        if self.args.internal_diagonse:
            return mae_normalized_list, mae_original_list
        else:
            return self.forecast_tensor_original