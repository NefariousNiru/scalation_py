from util.data_loading import data_provider
from util.tools import EarlyStopping, adjust_learning_rate, display_save_results, display_model_info
from util.QoF import diagnose
from util.plotting import plot_forecasts

from modeling.neuralmodeling.models import (
    Transformer, Informer, Autoformer, FEDformer, Crossformer, PatchTST,
    GPT4TS, LSTM_FFNN, Linear, DLinear, NLinear, FreTS, iTransformer,
    RLinear, LSTM_Seq2Seq, FFNN, Koopa, CycleiTransformer, CycleNet,
    PatchMixer, TimeMixer, TimeMixerPP, TimeBridge, TimeXer
)

from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import ray
from ray.air import session
from ray import tune

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
import warnings
import random
import datetime
from tqdm.notebook import tqdm

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, y_pred, y_true):
        mask = (y_true != 0.0)
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        mse_loss = torch.mean((y_true_masked - y_pred_masked) ** 2)
        return mse_loss

class Experiment(object):
    def __init__(self, args):
        if args is not None:
            self.args = args
            self.modeling_type = 'neural'

            self.normalization = args['normalization']

            self.skip_insample = self.args['skip_insample']
            self.target = self.args['target']
            self.dataset = self.args['dataset']
            self.normalization = self.args['normalization']


            self.same_n_samples = self.args['same_n_samples']
            self.plot_eda = self.args['plot_eda']

            self.debugging = self.args['debugging']
            self.modeling_approach = self.args['modeling_approach']
            self.epoch_gradient_tracker = {}

            self.forecast_type = self.args['forecast_type'].lower() if self.args['forecast_type'] is not None else None
            self.plot_scope_scale = self.args['plot_scope_scale'].lower() if self.args['plot_scope_scale'] is not None else self.args['plot_scope_scale']
            self.features = self.args['features'].lower() if self.args['features'] is not None else self.args['features']
            self.qof_calculation_mode = self.args['qof_calculation_mode'].lower() if self.args['qof_calculation_mode'] is not None else self.args['qof_calculation_mode']
            self.normalization = self.args['normalization'].lower() if self.args['normalization'] is not None else self.args['normalization']
            self.modeling_approach = self.args['modeling_approach'].lower() if self.args['modeling_approach'] is not None else self.args['modeling_approach']


            self.training_ratio = self.args['training_ratio']

            if self.training_ratio <= 0 or self.training_ratio >= 100:
                raise ValueError(
                    f"Invalid value for 'training_ratio'. Expected a fraction or whole number between 0 and 100. For 80% training ratio, both 0.8 and 80 should work.\n"
                    f"Received: {self.training_ratio}."
                )

            self.training_ratio = self.args['training_ratio'] / 100 if isinstance(self.args['training_ratio'],
                                                                               int) else self.args['training_ratio']

            if self.forecast_type not in ['point', 'interval']:
                raise ValueError(
                    f"Invalid value for 'forecast_type'. Expected one of the following 'point' or 'interval'\n"
                    f"Received: {self.forecast_type}."
                )

            if self.plot_scope_scale not in ['all_original', 'all_normalized', 'test_original', 'test_normalized',
                                             None]:
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

            if type(self.debugging) is not bool:
                raise ValueError(
                    f"Invalid value for 'self.debugging'. Expected a boolean (True or False)\n"
                    f"Received: {self.debugging}."
                )

            if type(self.plot_eda) is not bool:
                raise ValueError(
                    f"Invalid value for 'plot_eda'. Expected a boolean (True or False)\n"
                    f"Received: {self.plot_eda}."
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


            if self.skip_insample is None:
                self.validation = 'TrainNTest'
            else:
                self.validation = 'In-Sample'


            if self.skip_insample is None:
                None
            elif self.skip_insample < 0 or self.skip_insample == 0 or self.skip_insample >= len(self.data):
                raise ValueError(
                    f"Invalid value for 'skip_insample'. Expected one of the following:\n"
                    f"- A positive integer between 1 and {len(self.data)}.\n"
                    f"- None, to indicate TrainNTest.\n"
                    f"Received: {self.skip_insample}."
                )
            self.model_name = args['model_name']
            self.device = self._acquire_device()
            self.args['checkpoints'] = './checkpoints/'
            self.qof = None
            self.today = str(datetime.datetime.today().strftime('%Y-%m-%d'))

    def _acquire_device(self):
        if self.args['use_gpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args['gpu']) if not self.args['use_multi_gpu'] else self.args['devices']
            device = torch.device('cuda:{}'.format(self.args['gpu']))
            if self.debugging:
                print('Use GPU: cuda:{}'.format(self.args['gpu']))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        fix_seed = self.args['random_seed']
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'FEDformer': FEDformer,
            'Crossformer': Crossformer,
            'Transformer': Transformer,
            'PatchTST': PatchTST,
            'GPT4TS': GPT4TS,
            'LSTM_FFNN': LSTM_FFNN,
            'Linear': Linear,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'FreTS': FreTS,
            'iTransformer': iTransformer,
            'RLinear': RLinear,
            'LSTM_Seq2Seq': LSTM_Seq2Seq,
            'Koopa': Koopa,
            'FFNN': FFNN, 
            'CycleiTransformer': CycleiTransformer,
            'CycleNet': CycleNet,
            'PatchMixer': PatchMixer,
            'TimeMixer': TimeMixer,
            'TimeMixerPP': TimeMixerPP,
            'TimeBridge': TimeBridge,
            'TimeXer': TimeXer
        }
        model = model_dict[self.args['model_name']].Model(self.args).float()
        if self.args['use_multi_gpu'] and self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])

        return model.to(self.device)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self, self.args, flag)
        self.Data = data_set
        self.data = self.Data.data
        self.data_ = self.Data.data_
        self.train_size = self.Data.train_size
        self.test_size = self.Data.test_size
        self.columns = self.Data.columns
        self.target_feature = self.columns.get_loc(self.target)
        self.frequency = self.Data.frequency
        self.dates = self.Data.dates
        self.sample_mean = self.Data.sample_mean
        self.sample_mean_normalized = self.Data.sample_mean_normalized
        self.folder_path_plots = './plots/' + str(self.validation) + '/' + self.args['model_name'] + '/' + str(self.dataset) +'/'+str(self.pred_len)
        self.folder_path_results = './results/'+ str(self.validation) + '/'

        if not os.path.exists(self.folder_path_plots):
            os.makedirs(self.folder_path_plots)
        if not os.path.exists(self.folder_path_results):
            os.makedirs(self.folder_path_results)

        return data_set, data_loader, self.data, self.test_size

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        return model_optim

    def _select_criterion(self):
        criterion = MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (indices_x, indices_y, batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark, use_tf = False)
                else:
                    outputs = self.model(indices_x, indices_y, batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark, cycle_index, mode='vali', use_tf = False, count=-1)

                if isinstance(outputs, tuple):
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    else:
                        outputs = outputs[0], outputs[1]

                outputs = outputs[:, -self.args['pred_len']:, :]
                batch_y = batch_y[:, -self.args['pred_len']:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting) -> None:

        self.epoch_gradient = {}

        self.setting = setting

        self.args['pred_len'] = self.pred_len
        self.model = self._build_model()

        if self.args['debugging']:
            print(self.model)
            for name, param in self.model.named_parameters():
                print(name, param.requires_grad)

        self.total_params  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        train_data, train_loader, all_train, te_size = self._get_data(flag='train')
        vali_data, vali_loader, all_vali, te_size = self._get_data(flag='val')
        test_data, test_loader, all_test, te_size = self._get_data(flag='test')

        base_dir = self.args['cwd']
        self.path = os.path.join(base_dir, self.args['checkpoints'], setting)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args['patience'], verbose=self.debugging)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args['use_amp']:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                    steps_per_epoch = train_steps,
                                    pct_start = self.args['pct_start'],
                                    epochs = self.args['train_epochs'],
                                    max_lr = self.args['learning_rate'])
        train_losses = []
        vali_losses = []
        test_losses = []

        ref_var = 10
        num_vars = 12
        cos_logs = {j: [] for j in range(num_vars) if j != ref_var}
        ratio_logs = {j: [] for j in range(num_vars) if j != ref_var}
        grad_norms = []
        for epoch in tqdm(range(self.args['train_epochs'])):
            print('Epoch: ', epoch)
            epoch_gradients = []
            iter_count = 0
            train_loss = []
            epoch_loss = []

            self.model.train()
            epoch_time = time.time()
            count = 0
            for i, (indices_x, indices_y, batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(train_loader):
               
                iter_count += 1

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, trend_x_short, trend_x_long, batch_x_mark, dec_inp, batch_y, batch_y_mark, use_tf = self.args['use_tf'])

                        f_dim = -1 if self.features == 'MS' else 0
                        outputs = outputs[:, -self.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(indices_x, indices_y, batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark, cycle_index, mode='train', use_tf = self.args['use_tf'], count=-999)
                        
                    if isinstance(outputs, tuple):
                        if len(outputs) == 1:
                            outputs = outputs[0]
                        else:
                            outputs = outputs[0], outputs[1]

                if (i + 1) % 100 == 0:
                    if self.debugging:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args['train_epochs'] - epoch) * train_steps - i)
                    if self.debugging:
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args['use_amp']:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss = criterion(outputs, batch_y)
                    model_optim.zero_grad()
                    total_loss.backward()
                    model_optim.step()
                    epoch_loss.append(total_loss.item())

                if self.args['lradj'] == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            self.epoch_gradient[epoch] = epoch_gradients
            if self.debugging:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_losses.append(np.mean(epoch_loss))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            session.report({"vali_loss": vali_loss})

            vali_losses.append(vali_loss)
            test_losses.append(test_loss)
            if self.debugging:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, np.mean(epoch_loss), vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.path)

            if early_stopping.early_stop and self.args['debugging']:
                print("Early stopping")
                break
            
            if self.args['lradj'] != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=self.args['debugging'])
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
      
    def test(self, setting) -> None:

        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
        test_data, test_loader, all_test, te_size = self._get_data(flag='test')
        
        b_xs = []
        preds = []
        trues = []

        self.model.eval()
        count = 0 
        trend_tracker = 0
        print_params = None
        with torch.no_grad():
            
            for i, (indices_x, indices_y, batch_x, batch_y, batch_x_mark, batch_y_mark, cycle_index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
    
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
               
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(indices_x, indices_y, batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark, mode='test', use_tf=False)
                else:
                    outputs = self.model(indices_x, indices_y, batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark, cycle_index, mode='test', use_tf=False, count=count)

                if isinstance(outputs, tuple):
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    else:
                        outputs = outputs[0], outputs[1]
                
                if self.features == 'm':
                    outputs = outputs[:, -self.pred_len:, :]
                    batch_y = batch_y[:, -self.pred_len:, :].to(self.device)
                elif self.features == 'ms':
                    outputs = outputs[:, -self.pred_len:, :]
                    batch_y = batch_y[:, -self.pred_len:, :].to(self.device)
                elif self.features == 's':
                    outputs = outputs[:, -self.pred_len:, -1].unsqueeze(-1)
                    batch_y = batch_y[:, -self.pred_len:, -1].unsqueeze(-1).to(self.device)

                batch_x = batch_x.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                b_x = batch_x
                pred = outputs
                true = batch_y

                b_xs.append(b_x)
                preds.append(pred)
                trues.append(true)

        b_xs = np.concatenate(b_xs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        b_xs = b_xs.reshape(-1, b_xs.shape[-2], b_xs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        if self.debugging:
            print('test shape:', preds.shape, trues.shape)

        self.forecast_tensor = np.full(shape=(preds.shape[0], self.pred_len, self.args['c_out']),
                                       fill_value=np.nan)
        display_model_info(self)

        sample_offset = (self.pred_len - 1) if self.same_n_samples else 0
        for i in range(self.forecast_tensor.shape[-1]):
            for j in range(self.forecast_tensor.shape[0] - sample_offset):
                np.fill_diagonal(self.forecast_tensor[j:, :, i], preds[j, :, i])

        shape = self.forecast_tensor.shape
        self.forecast_tensor_original = test_data.inverse_transform(
            self.forecast_tensor.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
    def train_ray(self):

        if self.args['use_gpu'] and self.args['use_multi_gpu']:
            self.args['dvices'] = self.args['devices'].replace(' ', '')
            device_ids = self.args['devices'].split(',')
            self.args['device_ids'] = [int(id_) for id_ in device_ids]
            self.args['gpu'] = self.args['device_ids'][0]

        self.horizons = self.args['horizons']

        if type(self.horizons) is not list:
            raise ValueError(
                f"Invalid value for 'horizons'. Expected a list of target horizons.\n"
                f"Received: {self.horizons}."
            )

        if self.horizons != sorted(self.horizons):
            self.horizons.sort()

        if self.modeling_approach == 'joint':

            self.pred_len = max(self.horizons)

            self.modeling_approach = self.args['modeling_approach']

            setting = '{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
                self.args['model_id'], self.args['model_name'], self.args['data'], self.args['features'], self.args['seq_len'], self.args['label_len'],
                self.pred_len, self.args['d_model'], self.args['n_heads'], self.args['e_layers'], self.args['d_layers'], self.args['d_ff'],
                self.args['dropout'], self.args['learning_rate'])

            self.train(setting)

        elif self.modeling_approach == 'individual':
            self.modeling_approach = self.args['modeling_approach']
            for h in self.args['horizons']:
                self.horizons = [h]
                self.pred_len = h
                self.modeling_approach = self.args['modeling_approach']

                setting = '{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
                    self.args['model_id'], self.args['model_name'], self.args['data'], self.args['features'], self.args['seq_len'],
                    self.args['label_len'],
                    self.pred_len, self.args['d_model'], self.args['n_heads'], self.args['e_layers'], self.args['d_layers'],
                    self.args['d_ff'],
                    self.args['dropout'], self.args['learning_rate'])

                self.train(setting)

    def trainNtest(self):

        if self.args['use_gpu'] and self.args['use_multi_gpu']:
            self.args['dvices'] = self.args['devices'].replace(' ', '')
            device_ids = self.args['devices'].split(',')
            self.args['device_ids'] = [int(id_) for id_ in device_ids]
            self.args['gpu'] = self.args['device_ids'][0]

        self.horizons = self.args['horizons']

        if type(self.horizons) is not list:
            raise ValueError(
                f"Invalid value for 'horizons'. Expected a list of target horizons.\n"
                f"Received: {self.horizons}."
            )

        if self.horizons != sorted(self.horizons):
            self.horizons.sort()

        if self.modeling_approach == 'joint':
            self.qof = None
            self.pred_len = max(self.horizons)

            self.modeling_approach = self.args['modeling_approach']

            setting = '{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
                self.args['model_id'], self.args['model_name'], self.args['data'], self.args['features'], self.args['seq_len'], self.args['label_len'],
                self.pred_len, self.args['d_model'], self.args['n_heads'], self.args['e_layers'], self.args['d_layers'], self.args['d_ff'],
                self.args['dropout'], self.args['learning_rate'])

            self.train(setting)
            self.test(setting)

            if self.plot_scope_scale is not None:
                plot_forecasts(self)

            diagnose(self)
        elif self.modeling_approach == 'individual':
            self.qof = None
            self.modeling_approach = self.args['modeling_approach']
            for h in self.args['horizons']:
                self.horizons = [h]
                self.pred_len = h
                self.modeling_approach = self.args['modeling_approach']

                setting = '{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(
                    self.args['model_id'], self.args['model_name'], self.args['data'], self.args['features'], self.args['seq_len'],
                    self.args['label_len'],
                    self.pred_len, self.args['d_model'], self.args['n_heads'], self.args['e_layers'], self.args['d_layers'],
                    self.args['d_ff'],
                    self.args['dropout'], self.args['learning_rate'])

                self.train(setting)
                self.test(setting)

                if self.plot_scope_scale is not None:
                    plot_forecasts(self)
                self.args['mase_calc'] = None
                diagnose(self)
        display_save_results(self)

        return self.forecast_tensor_original