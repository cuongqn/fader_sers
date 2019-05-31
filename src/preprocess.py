# %%
import pandas as pd
import numpy as np
import os
import functools
from scipy.signal import butter, lfilter, filtfilt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SERSPreprocessor(object):
    def get_spectra_numpy(
        self,
        file_dir_list,
        y_list,
        norm='zscore', #standard, minmax
        sep="\t",
        header=None,
        remove_outliers=True,
        names=["x", "y", "wavenumber", "intensity"],
    ):
        load_func = functools.partial(
            self._preprocess_single_file, sep=sep, header=header, names=names
        )
        spectra_arr = list(map(load_func, file_dir_list))
        y_arr = [[y] * len(arr) for y, arr in zip(y_list,spectra_arr)]
        spectra_arr = np.concatenate(spectra_arr)
        y_arr = np.concatenate(y_arr)

        highpass_func = functools.partial(
            self._butter_bandpass_filter, lowcut=30, highcut=5000, fs=20001, order=1
        )
        spectra_arr = np.array(list(map(highpass_func, spectra_arr)))
        if remove_outliers:
            spectra_arr, y_arr = self._remove_outliers(spectra_arr, y_arr)
        if norm == 'zscore':
            spectra_arr = zscore(spectra_arr,axis=1)
        elif norm == 'standard':
            spectra_arr = StandardScaler().fit_transform(spectra_arr)
        elif norm == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(spectra_arr)
            spectra_arr = scaler.transform(spectra_arr)
        elif norm == 'global_minmax':
            spectra_arr = self._global_min_max_scaler(spectra_arr)
        
        return spectra_arr, y_arr
    
    @staticmethod
    def _remove_outliers(x, y):
        idx_out = np.where(zscore(x) > 7)
        idx_out = np.sort(np.unique(idx_out[0]))
        idx_keep = np.array([i for i in range(x.shape[0]) if i not in idx_out])
        x_clean = x[idx_keep]
        y_clean = y[idx_keep]
        return x_clean, y_clean

    @staticmethod
    def _global_min_max_scaler(x):
        min_ = x.min()
        max_ = x.max()
        delta = max_ - min_
        x_scaled = x - min_
        x_scaled = x_scaled/delta
        return x_scaled

    @staticmethod
    def _butter_bandpass_filter(data, highcut, lowcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, low, btype="highpass")
        data = filtfilt(b, a, data)
        b, a = butter(order, high, btype="lowpass")
        data = filtfilt(b, a, data)
        return data

    @staticmethod
    def _preprocess_single_file(file_dir, sep, header, names):
        data = pd.read_csv(file_dir, sep=sep, header=header, names=names)
        spectra_length = int(len(data) / data[names[-2]].value_counts().iloc[0])
        spectra_arr = data[names[-1]].values.reshape(-1, spectra_length)
        return spectra_arr

if __name__ == "__main__":
    data_dir = "data/raw/"
    y_list = [0, 10000]
    file_dir_list = [f"{data_dir}{conc}.txt" for conc in y_list]
    spectra_arr, y_arr = SERSPreprocessor().get_spectra_numpy(file_dir_list, y_list = y_list)


# #%%
# import matplotlib.pyplot as plt
# # plt.plot(spectra_arr[0])
# plt.plot(spectra_arr[400])
# plt.show()


#%%
