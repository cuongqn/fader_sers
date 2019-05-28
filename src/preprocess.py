# %%
import pandas as pd
import numpy as np
import os
import functools
from scipy.signal import butter, lfilter, filtfilt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

class SERSPreprocessor(object):
    def get_spectra_numpy(
        self,
        file_dir_list,
        norm='spectra',
        sep="\t",
        header=None,
        names=["x", "y", "wavenumber", "intensity"],
    ):
        load_func = functools.partial(
            self._preprocess_single_file, sep=sep, header=header, names=names
        )
        spectra_arr = list(map(load_func, file_dir_list))
        spectra_arr = np.concatenate(spectra_arr)

        highpass_func = functools.partial(
            self.butter_bandpass_filter, lowcut=30, highcut=5000, fs=20001, order=1
        )
        spectra_arr = np.array(list(map(highpass_func, spectra_arr)))
        
        if norm == 'spectra':
            spectra_arr = zscore(spectra_arr,axis=1)
        elif norm == 'feature':
            spectra_arr = StandardScaler().fit_transform(spectra_arr)
        return spectra_arr

    @staticmethod
    def butter_bandpass_filter(data, highcut, lowcut, fs, order=5):
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
    file_dir_list = [data_dir + conc for conc in ["0.txt", "100000.txt"]]
    spectra_arr = SERSPreprocessor().get_spectra_numpy(file_dir_list)


# #%%
# import matplotlib.pyplot as plt
# # plt.plot(spectra_arr[0])
# plt.plot(spectra_arr[400])
# plt.show()

# #%%
# spectra_arr.shape

#%%
