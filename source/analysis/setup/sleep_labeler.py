import numpy as np

from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.sleep_label import SleepWakeLabel
from source.analysis.setup.sleep_label import ThreeClassLabel


class SleepLabeler(object):

    @staticmethod
    def label_sleep_wake(raw_sleep_wake):
        labeled_sleep = []

        for value in raw_sleep_wake:
            if value > 0:
                converted_value = SleepWakeLabel.sleep.value
            else:
                converted_value = SleepWakeLabel.wake.value
            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def label_three_class(raw_sleep_wake):
        labeled_sleep = []

        for value in raw_sleep_wake:
            if value == 0:
                converted_value = ThreeClassLabel.wake.value
            elif value == 5:
                converted_value = ThreeClassLabel.rem.value
            else:
                converted_value = ThreeClassLabel.nrem.value

            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def label_one_vs_rest(sleep_wake_labels, positive_class):
        labeled_sleep = []

        for value in sleep_wake_labels:
            if value == positive_class:
                converted_value = 1
            else:
                converted_value = 0

            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def convert_three_class_to_two(raw_performance: RawPerformance):
        raw_performance.true_labels = SleepLabeler.label_sleep_wake(raw_performance.true_labels)
        number_of_samples = np.shape(raw_performance.class_probabilities)[0]
        for index in range(number_of_samples):
            raw_performance.class_probabilities[index, 1] = raw_performance.class_probabilities[index, 1] + \
                                                            raw_performance.class_probabilities[index, 2]
        raw_performance.class_probabilities = raw_performance.class_probabilities[:, :-1]

        return raw_performance
