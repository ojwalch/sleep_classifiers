import random

import numpy as np

from source.analysis.setup.data_split import DataSplit


class TrainTestSplitter(object):

    @staticmethod
    def leave_one_out(subject_ids):
        splits = []

        for index in range(len(subject_ids)):
            training_set = subject_ids.copy()
            testing_set = [training_set.pop(index)]

            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))

        return splits

    @staticmethod
    def by_fraction(subject_ids, test_fraction, number_of_splits):

        test_index = int(np.round(test_fraction * len(subject_ids)))

        splits = []
        for trial in range(number_of_splits):
            random.shuffle(subject_ids)

            training_set = subject_ids.copy()
            testing_set = []
            for index in range(test_index):
                testing_set.append(training_set.pop(0))

            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))

        return splits
