import collections as co
import numpy as np
import os
import pathlib

import torch
import urllib.request
import zipfile
import time_dataset
import common
import pandas as pd 
from sklearn import preprocessing
here = pathlib.Path(__file__).resolve().parent

DATA_PATH = os.path.dirname(os.path.abspath(__file__))

def _pad(channel, maxlen):
    # X중 하나의 데이터 들어옴 (Series) - (116,)
    channel = torch.tensor(channel) # Series를 tensor로 바꿈
    out = torch.full((maxlen,), channel[-1]) # tensor의 마지막 원소를 (maxlen,) 크기로 새로운 텐서 out을 만듦
    out[:channel.size(0)] = channel # 텐서 out의 원래 범위 만큼을 원래 값으로 채움
    return out # 리턴


def _process_data(look_window, forecast_window, stride_window, missing_rate, loc, learning_method):
    PATH = os.path.dirname(os.path.abspath(__file__))
    from sklearn import preprocessing
    torch.__version__

    data_path = PATH + '/data'
    # Reading back and setting the MultiIndex
    labelled_data = pd.read_feather('../MTDataBatch1\Labelled Data\Labelled Batch 1 - Updated.feather')
    labelled_data.set_index(['DeviceId', 'time'], inplace=True)
    labelled_data = labelled_data.sort_index(level=['DeviceId', 'time'])
    column_list = list(labelled_data.columns)
    train_size = int(len(labelled_data) * 0.80)
    test_size = len(labelled_data) - train_size
    # Training data : take all rows, and only device feature columns (From "sensors.battery.extpower" to "power_difference")
    train_data = labelled_data.iloc[0:train_size, 6:29]
    test_data = labelled_data.iloc[train_size:len(labelled_data), 6:29]
    # Drop columns that have string values, or even NaN values
    # Manual Inspection proved them as non important for malfunction detectio.
    train_data = train_data.drop('location.source', axis=1)
    train_data = train_data.drop('location.numSats', axis=1)
    train_data = train_data.drop('sensors.accelerometer.y', axis=1)
    train_data = train_data.drop('sensors.accelerometer.x', axis=1)
    train_data = train_data.drop('sensors.accelerometer.z', axis=1)
    test_data = test_data.drop('location.source', axis=1)
    test_data = test_data.drop('location.numSats', axis=1)
    test_data = test_data.drop('sensors.accelerometer.y', axis=1)
    test_data = test_data.drop('sensors.accelerometer.x', axis=1)
    test_data = test_data.drop('sensors.accelerometer.z', axis=1)
    # Labels is the array of values in "isDeviceWorking" -  convert to array of inetgers (0 or 1 values)
    labels = labelled_data.iloc[:, 29].values
    deviceWorkingLabels = labelled_data[['isDeviceWorking']].copy()
    deviceWorkingLabels['isDeviceWorking'] = deviceWorkingLabels['isDeviceWorking'].astype(int)
    train_data_label = deviceWorkingLabels.iloc[0:train_size, :]
    test_data_label = deviceWorkingLabels.iloc[train_size:len(deviceWorkingLabels), :]
    # Scale column values (without normalizing the indexes)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Group by level 0, which is the first index corresponding to 'DeviceId'
    # After applying the scaling function (which output a Numpy array), convert it to a dataframe

    normalized_train_data = train_data.groupby(level=0).apply(
        lambda x: pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index).round(5))

    normalized_test_data = test_data.groupby(level=0).apply(
        lambda x: pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index).round(5))
    testdeviceIDs = set(normalized_test_data.index.get_level_values(0).tolist())
    X_times = normalized_train_data.to_numpy()
    y_train_label = train_data_label.to_numpy()
    X_times_test = normalized_test_data.to_numpy()
    y_label = test_data_label.to_numpy()
    input_size = X_times.shape[1]
    # X_times = time_dataset.normalize(X_times)
    # X_times_test = time_dataset.normalize(X_times_test)
    y_train_label = y_train_label[:X_times.shape[0]]
    total_length = len(X_times)

    timelen = X_times.shape[0]
    timelen_test = X_times_test.shape[0]

    full_seq_data_train = torch.Tensor()
    forecast_seq_train = torch.Tensor()
    full_y_seq_data_train = torch.Tensor()
    full_forecast_data_train = torch.Tensor()
    full_seq_data_test = torch.Tensor()
    full_y_seq_data = torch.Tensor()
    full_forecast_data_test = torch.Tensor()
    forecast_seq_test = torch.Tensor()

    # TRAIN
    for _len in range(int((timelen - look_window + stride_window - forecast_window) / stride_window)):
        full_seq_temp = torch.Tensor([X_times[(_len * stride_window):(_len * stride_window) + look_window]])
        forecast_seq_temp = torch.Tensor(
            [X_times[(_len * stride_window) + look_window:(_len * stride_window) + look_window + forecast_window]])

        full_y_seq_temp = torch.Tensor([y_train_label[(_len * stride_window):(_len * stride_window) + look_window]])
        forecast_y_seq_temp = torch.Tensor([y_train_label[(_len * stride_window) + look_window:(
                                                                                                           _len * stride_window) + look_window + forecast_window]])

        full_seq_data_train = torch.cat([full_seq_data_train, full_seq_temp])
        forecast_seq_train = torch.cat([forecast_seq_train, forecast_seq_temp])
        full_y_seq_data_train = torch.cat([full_y_seq_data_train, full_y_seq_temp])
        full_forecast_data_train = torch.cat([full_forecast_data_train, forecast_y_seq_temp])

    if missing_rate == 0:

        DATA_PATH_SAVE = PATH + '/processed_data/look_' + str(look_window) + '_stride_' + str(
            stride_window) + '_forecast_' + str(forecast_window)
    else:
        DATA_PATH_SAVE = PATH + '/processed_data/look_' + str(look_window) + '_stride_' + str(
            stride_window) + '_forecast_' + str(forecast_window) + '_Missing_' + str(missing_rate)

    # TEST
    for _len in range(int((timelen_test - look_window + stride_window - forecast_window) / stride_window)):
        # import pdb ; pdb.set_trace()
        full_seq_temp = torch.Tensor([X_times_test[(_len * stride_window):(_len * stride_window) + look_window]])
        forecast_seq_temp = torch.Tensor(
            [X_times_test[(_len * stride_window) + look_window:(_len * stride_window) + look_window + forecast_window]])

        full_y_seq_temp = torch.Tensor([y_label[(_len * stride_window):(_len * stride_window) + look_window]])
        forecast_y_seq_temp = torch.Tensor(
            [y_label[(_len * stride_window) + look_window:(_len * stride_window) + look_window + forecast_window]])

        full_seq_data_test = torch.cat([full_seq_data_test, full_seq_temp])
        forecast_seq_test = torch.cat([forecast_seq_test, forecast_seq_temp])
        full_y_seq_data = torch.cat([full_y_seq_data, full_y_seq_temp])
        full_forecast_data_test = torch.cat([full_forecast_data_test, forecast_y_seq_temp])
    # import pdb ; pdb.set_trace()
    if missing_rate != 0:
        generator = torch.Generator().manual_seed(56789)
        # import pdb ; pdb.set_trace()
        for Xi in full_seq_data_train:
            removed_points = torch.randperm(full_seq_data_train.size(1), generator=generator)[
                             :int(full_seq_data_train.size(1) * missing_rate)].sort().values
            Xi[removed_points] = float('nan')
        for Xi in full_seq_data_test:
            removed_points = torch.randperm(full_seq_data_test.size(1), generator=generator)[
                             :int(full_seq_data_test.size(1) * missing_rate)].sort().values
            Xi[removed_points] = float('nan')

    eval_length = full_seq_data_test.shape[0]

    train_seq_data = full_seq_data_train
    train_y_data = full_y_seq_data_train
    train_forecast_data = full_forecast_data_train
    train_forecast_seq = forecast_seq_train

    val_seq_data = full_seq_data_test[:int(eval_length * 0.5)]
    val_y_data = full_y_seq_data[:int(eval_length * 0.5)]
    val_forecast_data = full_forecast_data_test[:int(eval_length * 0.5)]
    val_forecast_seq = forecast_seq_test[:int(eval_length * 0.5)]

    test_seq_data = full_seq_data_test[int(eval_length * 0.5):]
    test_y_data = full_y_seq_data[int(eval_length * 0.5):]
    test_forecast_data = full_forecast_data_test[int(eval_length * 0.5):]
    test_forecast_seq = forecast_seq_test[int(eval_length * 0.5):]

    torch.save(train_seq_data, DATA_PATH_SAVE + '/train_seq_data.pt')
    torch.save(train_y_data, DATA_PATH_SAVE + '/train_y_data.pt')
    torch.save(train_forecast_data, DATA_PATH_SAVE + '/train_forecast_y.pt')
    torch.save(train_forecast_seq, DATA_PATH_SAVE + '/train_forecast_seq.pt')

    torch.save(val_seq_data, DATA_PATH_SAVE + '/val_seq_data.pt')
    torch.save(val_y_data, DATA_PATH_SAVE + '/val_y_data.pt')
    torch.save(val_forecast_data, DATA_PATH_SAVE + '/val_forecast_y.pt')
    torch.save(val_forecast_seq, DATA_PATH_SAVE + '/val_forecast_seq.pt')

    torch.save(test_seq_data, DATA_PATH_SAVE + '/test_seq_data.pt')
    torch.save(test_y_data, DATA_PATH_SAVE + '/test_y_data.pt')
    torch.save(test_forecast_data, DATA_PATH_SAVE + '/test_forecast_y.pt')
    torch.save(test_forecast_seq, DATA_PATH_SAVE + '/test_forecast_seq.pt')

    times = torch.Tensor(np.arange(look_window))
    torch.save(times, DATA_PATH_SAVE + '/times.pt')
    print("DONE")
    return input_size


def get_data(look_window, forecast_window, stride_window, missing_rate, learning_method='self-supervised'):
    # Reading back and setting the MultiIndex
    labelled_data = pd.read_feather('../MTDataBatch1\Labelled Data\Labelled Batch 1 - Updated.feather')
    labelled_data.set_index(['DeviceId', 'time'], inplace=True)
    labelled_data = labelled_data.sort_index(level=['DeviceId', 'time'])
    column_list = list(labelled_data.columns)
    train_size = int(len(labelled_data) * 0.80)
    test_size = len(labelled_data) - train_size
    # Training data : take all rows, and only device feature columns (From "sensors.battery.extpower" to "power_difference")
    train_data = labelled_data.iloc[0:train_size, 6:29]
    test_data = labelled_data.iloc[train_size:len(labelled_data), 6:29]
    # Drop columns that have string values, or even NaN values
    # Manual Inspection proved them as non important for malfunction detectio.
    train_data = train_data.drop('location.source', axis=1)
    train_data = train_data.drop('location.numSats', axis=1)
    train_data = train_data.drop('sensors.accelerometer.y', axis=1)
    train_data = train_data.drop('sensors.accelerometer.x', axis=1)
    train_data = train_data.drop('sensors.accelerometer.z', axis=1)
    test_data = test_data.drop('location.source', axis=1)
    test_data = test_data.drop('location.numSats', axis=1)
    test_data = test_data.drop('sensors.accelerometer.y', axis=1)
    test_data = test_data.drop('sensors.accelerometer.x', axis=1)
    test_data = test_data.drop('sensors.accelerometer.z', axis=1)
    # Labels is the array of values in "isDeviceWorking" -  convert to array of inetgers (0 or 1 values)
    labels = labelled_data.iloc[:, 29].values
    deviceWorkingLabels = labelled_data[['isDeviceWorking']].copy()
    deviceWorkingLabels['isDeviceWorking'] = deviceWorkingLabels['isDeviceWorking'].astype(int)
    train_data_label = deviceWorkingLabels.iloc[0:train_size, :]
    test_data_label = deviceWorkingLabels.iloc[train_size:len(deviceWorkingLabels), :]
    # Scale column values (without normalizing the indexes)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Group by level 0, which is the first index corresponding to 'DeviceId'
    # After applying the scaling function (which output a Numpy array), convert it to a dataframe

    normalized_train_data = train_data.groupby(level=0).apply(
        lambda x: pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index).round(5))

    normalized_test_data = test_data.groupby(level=0).apply(
        lambda x: pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index).round(5))
    testdeviceIDs = set(normalized_test_data.index.get_level_values(0).tolist())
    
    base_base_loc = here / 'processed_data'
    if learning_method == 'unsupervised':
        if missing_rate == 0:
            loc = base_base_loc / ('look_' + str(look_window) + '_stride_' + str(stride_window) + '_forecast_' + str(
                forecast_window))
        # loc = base_base_loc / ('EXPSMD_121_look_'+str(look_window)+'_stride_'+str(stride_window)+'_Missing_'+str(missing_rate))
        else:
            loc = base_base_loc / ('look_' + str(look_window) + '_stride_' + str(stride_window) + '_forecast_' + str(
                forecast_window) + '_Missing_' + str(missing_rate))

    else:
        if missing_rate == 0:
            loc = base_base_loc / ('look_' + str(look_window) + '_stride_' + str(stride_window) + '_forecast_' + str(
                forecast_window))
        # loc = base_base_loc / ('EXPSMD_121_look_'+str(look_window)+'_stride_'+str(stride_window)+'_Missing_'+str(missing_rate))
        else:
            loc = base_base_loc / ('look_' + str(look_window) + '_stride_' + str(stride_window) + '_forecast_' + str(
                forecast_window) + '_Missing_' + str(missing_rate))
    if os.path.exists(loc):
        PATH = os.path.dirname(os.path.abspath(__file__))
        data_path = PATH + '/data'
        # import pdb ; pdb.set_trace()
        X_times = normalized_train_data.to_numpy()
        input_size = X_times.shape[1]
        # pass
    else:
        # download()

        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        input_size = _process_data(look_window, forecast_window, stride_window, missing_rate, loc, learning_method)

    return loc, input_size