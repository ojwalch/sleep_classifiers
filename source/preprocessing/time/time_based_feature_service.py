import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants


class TimeBasedFeatureService(object):
    @staticmethod
    def load_time(subject_id):
        feature_path = TimeBasedFeatureService.get_path_for_time(subject_id)
        feature = pd.read_csv(str(feature_path)).values
        return feature

    @staticmethod
    def get_path_for_time(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_time_feature.out')

    @staticmethod
    def write_time(subject_id, feature):
        feature_path = TimeBasedFeatureService.get_path_for_time(subject_id)
        np.savetxt(feature_path, feature, fmt='%f')

    @staticmethod
    def load_circadian_model(subject_id):
        feature_path = TimeBasedFeatureService.get_path_for_circadian_model(subject_id)
        feature = pd.read_csv(str(feature_path), delimiter=' ').values
        return feature

    @staticmethod
    def get_path_for_circadian_model(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_circadian_feature.out')

    @staticmethod
    def write_circadian_model(subject_id, feature):
        feature_path = TimeBasedFeatureService.get_path_for_circadian_model(subject_id)
        np.savetxt(feature_path, feature, fmt='%f')

    @staticmethod
    def load_cosine(subject_id):
        feature_path = TimeBasedFeatureService.get_path_for_cosine(subject_id)
        feature = pd.read_csv(str(feature_path)).values
        return feature

    @staticmethod
    def get_path_for_cosine(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_cosine_feature.out')

    @staticmethod
    def write_cosine(subject_id, feature):
        feature_path = TimeBasedFeatureService.get_path_for_cosine(subject_id)
        np.savetxt(feature_path, feature, fmt='%f')

    @staticmethod
    def build_time(valid_epochs):
        features = []
        first_timestamp = valid_epochs[0].timestamp
        for epoch in valid_epochs:
            value = epoch.timestamp - first_timestamp

            value = value / 3600.0  # Changing units to hours improves performance

            features.append(value)
        return np.array(features)

    @staticmethod
    def build_circadian_model(subject_id, valid_epochs):
        circadian_file = utils.get_project_root().joinpath('data/circadian_predictions/' + subject_id +
                                                           '_clock_proxy.txt')
        if circadian_file.is_file():
            circadian_model = pd.read_csv(str(circadian_file), delimiter=',').values

            return TimeBasedFeatureService.build_circadian_model_from_raw(circadian_model, valid_epochs)

    @staticmethod
    def cosine_proxy(time):
        sleep_drive_cosine_shift = 5
        return -1 * np.math.cos((time - sleep_drive_cosine_shift * Constants.SECONDS_PER_HOUR) *
                                2 * np.math.pi / Constants.SECONDS_PER_DAY)

    @staticmethod
    def build_cosine(valid_epochs):
        features = []
        first_value = TimeBasedFeatureService.cosine_proxy(0)
        first_timestamp = valid_epochs[0].timestamp

        for epoch in valid_epochs:
            value = TimeBasedFeatureService.cosine_proxy(epoch.timestamp - first_timestamp)
            normalized_value = value
            features.append(normalized_value)

        return np.array(features)

    @staticmethod
    def build_circadian_model_from_raw(circadian_model, valid_epochs):
        features = []

        first_inactive_epoch = valid_epochs[0]
        first_value = np.interp(first_inactive_epoch.timestamp, circadian_model[:, 0], circadian_model[:, 1])

        for epoch in valid_epochs:
            time = epoch.timestamp
            value = np.interp(time, circadian_model[:, 0], circadian_model[:, 1])
            normalized_value = (value - first_value) / (np.amin((circadian_model[:, 1] - first_value)))

            if normalized_value < Constants.LOWER_BOUND:
                normalized_value = Constants.LOWER_BOUND

            features.append([normalized_value])

        feature_array = np.array(features)

        return feature_array
