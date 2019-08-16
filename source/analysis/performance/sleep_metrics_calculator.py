import numpy as np

from source.constants import Constants


class SleepMetricsCalculator(object):

    @staticmethod
    def get_tst(labels):
        sleep_epoch_indices = np.where(labels > 0)
        tst = np.shape(sleep_epoch_indices)[1]
        return tst * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE

    @staticmethod
    def get_wake_after_sleep_onset(labels):
        sleep_indices = np.argwhere(labels > 0)
        if np.shape(sleep_indices)[0] > 0:
            sol_index = np.amin(sleep_indices)
            indices_where_wake_occurred = np.where(labels == 0)

            waso_indices = np.where(indices_where_wake_occurred > sol_index)
            waso_indices = waso_indices[1]
            number_waso_indices = np.shape(waso_indices)[0]
            return number_waso_indices * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE
        else:
            return len(labels) * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE

    @staticmethod
    def get_sleep_efficiency(labels):
        sleep_indices = np.where(labels > 0)
        sleep_efficiency = float(np.shape(sleep_indices)[1]) / float(np.shape(labels)[0])
        return sleep_efficiency

    @staticmethod
    def get_sleep_onset_latency(labels):
        sleep_indices = np.argwhere(labels > 0)
        if np.shape(sleep_indices)[0] > 0:
            return np.amin(sleep_indices) * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE
        else:
            return len(labels) * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE

    @staticmethod
    def get_time_in_rem(labels):
        rem_epoch_indices = np.where(labels == 2)
        rem_time = np.shape(rem_epoch_indices)[1]
        return rem_time * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE

    @staticmethod
    def get_time_in_nrem(labels):
        rem_epoch_indices = np.where(labels == 1)
        rem_time = np.shape(rem_epoch_indices)[1]
        return rem_time * Constants.EPOCH_DURATION_IN_SECONDS / Constants.SECONDS_PER_MINUTE
