import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch


class ActivityCountFeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15

    @staticmethod
    def load(subject_id):
        activity_count_feature_path = ActivityCountFeatureService.get_path(subject_id)
        feature = pd.read_csv(str(activity_count_feature_path)).values
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_count_feature.out')

    @staticmethod
    def write(subject_id, feature):
        activity_counts_feature_path = ActivityCountFeatureService.get_path(subject_id)
        np.savetxt(activity_counts_feature_path, feature, fmt='%f')

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - ActivityCountFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + ActivityCountFeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def build(subject_id, valid_epochs):
        activity_count_collection = ActivityCountService.load_cropped(subject_id)
        return ActivityCountFeatureService.build_from_collection(activity_count_collection, valid_epochs)

    @staticmethod
    def build_from_collection(activity_count_collection, valid_epochs):
        count_features = []

        interpolated_timestamps, interpolated_counts = ActivityCountFeatureService.interpolate(
            activity_count_collection)

        for epoch in valid_epochs:
            indices_in_range = ActivityCountFeatureService.get_window(interpolated_timestamps, epoch)
            activity_counts_in_range = interpolated_counts[indices_in_range]

            feature = ActivityCountFeatureService.get_feature(activity_counts_in_range)
            count_features.append(feature)

        return np.array(count_features)

    @staticmethod
    def get_feature(count_values):
        convolution = utils.smooth_gauss(count_values.flatten(), np.shape(count_values.flatten())[0])
        return np.array([convolution])

    @staticmethod
    def interpolate(activity_count_collection):
        timestamps = activity_count_collection.timestamps.flatten()
        activity_count_values = activity_count_collection.values.flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_counts = np.interp(interpolated_timestamps, timestamps, activity_count_values)
        return interpolated_timestamps, interpolated_counts
