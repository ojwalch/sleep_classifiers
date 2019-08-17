import os

import pandas as pd

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
    def crop(activity_count_collection, interval):
        subject_id = activity_count_collection.subject_id
        timestamps = activity_count_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = activity_count_collection.data[valid_indices, :]
        return ActivityCountCollection(subject_id=subject_id, data=cropped_data)

