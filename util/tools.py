import numpy as np
import torch
from tabulate import tabulate
import pandas as pd

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args['learning_rate'] * (0.2 ** (epoch // 2))
    if args['lradj'] == 'type1':
        lr_adjust = {epoch: args['learning_rate'] * (0.5 ** ((epoch - 1) // 1))}
    elif args['lradj'] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args['lradj'] == 'type3':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 3 else args['learning_rate'] * (0.9 ** ((epoch - 3) // 1))}
    elif args['lradj'] == 'constant':
        lr_adjust = {epoch: args['learning_rate']}
    elif args['lradj'] == '3':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 10 else args['learning_rate']*0.1}
    elif args['lradj'] == '4':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 15 else args['learning_rate']*0.1}
    elif args['lradj'] == '5':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 25 else args['learning_rate']*0.1}
    elif args['lradj'] == '6':
        lr_adjust = {epoch: args['learning_rate'] if epoch < 5 else args['learning_rate']*0.1}
    elif args['lradj'] == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        #self.val_loss_min = np.Inf
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def display_save_results(self):

    file_path = str('results/results '+self.today+'.csv')
    averages = ['Avg.', ''] + self.qof.iloc[:, 2:].mean().tolist()
    self.qof.loc[len(self.qof)] = averages
    target = self.args['target']
    print(f'\033[1mQuality of Fit (QoF) for {self.model_name} Task: {self.features.upper()} Target: {target}\033[0m')
    columns_to_display = self.args['qof_to_display']
    print(tabulate(self.qof[columns_to_display].round(4), headers='keys', tablefmt='pretty', showindex=False))
    with open(file_path, 'a', newline='') as file:
        file.write('\n' + self.model_name + ' ' + self.validation + '\n')
        self.qof[columns_to_display].to_csv(file, header=True, index=False, float_format='%.3f')

def display_model_info(self):
    model_info = [
        ["Model name", self.model_name],
        ["Target", self.target],
        ["Forecasting horizons", self.pred_len],
        ["Prediction features", self.features.upper()],
        ["Normalization", self.normalization],
        ["Data shape", self.data.shape],
        ["Train size", self.train_size],
        ["Test size", self.test_size],
        ["Start date", self.dates.iloc[0]],
        ["End date", self.dates.iloc[-1]],
        ["Forecast tensor shape", self.forecast_tensor.shape],
        ["QoF calculation mode", self.qof_calculation_mode],
        ["Validation", self.validation],
        ["Total params", self.total_params],
    ]
    print(tabulate(model_info, headers=["Model Info.", ""], tablefmt='pretty'))
    print('')


