import numpy as np
import pandas as pd

from source.constants import Constants


class MotionFeatureService(object):

    @staticmethod
    def load(subject_id):
        motion_feature_path = MotionFeatureService.get_path(subject_id)
        feature = pd.read_csv(str(motion_feature_path)).values
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_motion_feature.out')

    @staticmethod
    def write(subject_id, feature):
        motion_feature_path = MotionFeatureService.get_path(subject_id)
        np.savetxt(motion_feature_path, feature, fmt='%f')
