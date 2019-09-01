import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from source import utils
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_collection import ActivityCountCollection


class ActivityCountService(object):
    @staticmethod
    def load_cropped(subject_id):
        activity_counts_path = ActivityCountService.get_cropped_file_path(subject_id)
        counts_array = ActivityCountService.load(activity_counts_path)
        return ActivityCountCollection(subject_id=subject_id, data=counts_array)

    @staticmethod
    def load(counts_file):
        counts_array = pd.read_csv(str(counts_file)).values
        return counts_array

    @staticmethod
    def get_cropped_file_path(subject_id):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_counts.out")

    @staticmethod
    def build_activity_counts():
        os.system(Constants.MATLAB_PATH + ' -nodisplay -nosplash -nodesktop -r \"run(\'' + str(
            utils.get_project_root()) + '/source/make_counts.m\'); exit;\"')

    @staticmethod
    def build_activity_counts_without_matlab(subject_id, data):

        fs = 50
        time = np.arange(np.amin(data[:, 0]), np.amax(data[:, 0]), 1.0 / fs)
        z_data = np.interp(time, data[:, 0], data[:, 3])

        cf_low = 3
        cf_hi = 11
        order = 5
        w1 = cf_low / (fs / 2)
        w2 = cf_hi / (fs / 2)
        pass_band = [w1, w2]
        b, a = butter(order, pass_band, 'bandpass')

        z_filt = filtfilt(b, a, z_data)
        z_filt = np.abs(z_filt)
        top_edge = 5
        bottom_edge = 0
        number_of_bins = 128

        bin_edges = np.linspace(bottom_edge, top_edge, number_of_bins + 1)
        binned = np.digitize(z_filt, bin_edges)
        epoch = 15
        counts = ActivityCountService.max2epochs(binned, fs, epoch)
        counts = (counts - 18) * 3.07
        counts[counts < 0] = 0

        time_counts = np.linspace(np.min(data[:, 0]), max(data[:, 0]), np.shape(counts)[0])
        time_counts = np.expand_dims(time_counts, axis=1)
        counts = np.expand_dims(counts, axis=1)
        output = np.hstack((time_counts, counts))

        activity_count_output_path = ActivityCountService.get_cropped_file_path(subject_id)
        np.savetxt(activity_count_output_path, output, fmt='%f', delimiter=',')

    @staticmethod
    def max2epochs(data, fs, epoch):
        data = data.flatten()

        seconds = int(np.floor(np.shape(data)[0] / fs))
        data = np.abs(data)
        data = data[0:int(seconds * fs)]

        data = data.reshape(fs, seconds, order='F').copy()

        data = data.max(0)
        data = data.flatten()
        N = np.shape(data)[0]
        num_epochs = int(np.floor(N / epoch))
        data = data[0:(num_epochs * epoch)]

        data = data.reshape(epoch, num_epochs, order='F').copy()
        epoch_data = np.sum(data, axis=0)
        epoch_data = epoch_data.flatten()

        return epoch_data

    @staticmethod
    def crop(activity_count_collection, interval):
        subject_id = activity_count_collection.subject_id
        timestamps = activity_count_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = activity_count_collection.data[valid_indices, :]
        return ActivityCountCollection(subject_id=subject_id, data=cropped_data)
