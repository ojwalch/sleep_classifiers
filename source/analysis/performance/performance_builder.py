import numpy as np
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, accuracy_score, recall_score, precision_score

from source.analysis.performance.epoch_performance import SleepWakePerformance
from source.analysis.setup.sleep_label import SleepWakeLabel, ThreeClassLabel
from source.analysis.setup.sleep_labeler import SleepLabeler


class PerformanceBuilder(object):

    @staticmethod
    def build_with_sleep_threshold(raw_performance, sleep_threshold):

        if np.shape(raw_performance.class_probabilities)[1] > 2:
            raw_performance = SleepLabeler.convert_three_class_to_two(raw_performance)

        false_positive_rates, true_positive_rates, thresholds = roc_curve(raw_performance.true_labels,
                                                                          raw_performance.class_probabilities[:,
                                                                          SleepWakeLabel.sleep.value],
                                                                          pos_label=SleepWakeLabel.sleep.value,
                                                                          drop_intermediate=False)
        auc_value = auc(false_positive_rates, true_positive_rates)

        predicted_labels = PerformanceBuilder.apply_threshold_sleep_wake(raw_performance, sleep_threshold)

        kappa = cohen_kappa_score(raw_performance.true_labels, predicted_labels)

        accuracy = accuracy_score(raw_performance.true_labels, predicted_labels)
        wake_correct = recall_score(raw_performance.true_labels, predicted_labels, pos_label=SleepWakeLabel.wake.value)
        sleep_correct = recall_score(raw_performance.true_labels, predicted_labels,
                                     pos_label=SleepWakeLabel.sleep.value)
        sleep_predictive_value = precision_score(raw_performance.true_labels, predicted_labels,
                                                 pos_label=SleepWakeLabel.sleep.value)
        wake_predictive_value = precision_score(raw_performance.true_labels, predicted_labels,
                                                pos_label=SleepWakeLabel.wake.value)

        return SleepWakePerformance(accuracy=accuracy,
                                    wake_correct=wake_correct,
                                    sleep_correct=sleep_correct,
                                    kappa=kappa,
                                    auc=auc_value,
                                    sleep_predictive_value=sleep_predictive_value,
                                    wake_predictive_value=wake_predictive_value)

    @staticmethod
    def build_with_true_positive_rate_threshold(raw_performance, true_positive_threshold):
        false_positive_rates, true_positive_rates, thresholds = \
            roc_curve(raw_performance.true_labels,
                      raw_performance.class_probabilities[:, SleepWakeLabel.sleep.value],
                      pos_label=SleepWakeLabel.sleep.value,
                      drop_intermediate=False)

        sleep_threshold = np.interp(true_positive_threshold, true_positive_rates, thresholds)

        performance = PerformanceBuilder.build_with_sleep_threshold(raw_performance, sleep_threshold)

        return performance

    @staticmethod
    def apply_threshold_sleep_wake(raw_performance, sleep_threshold):
        predicted_labels = []

        number_of_samples = np.shape(raw_performance.class_probabilities)[0]
        for index in range(number_of_samples):
            if raw_performance.class_probabilities[index, 1] >= sleep_threshold:
                predicted_labels.append(SleepWakeLabel.sleep.value)
            else:
                predicted_labels.append(SleepWakeLabel.wake.value)

        return np.array(predicted_labels)

    @staticmethod
    def apply_threshold_three_class(raw_performance, wake_threshold, rem_threshold):
        predicted_labels = []

        number_of_samples = np.shape(raw_performance.class_probabilities)[0]
        for index in range(number_of_samples):
            if raw_performance.class_probabilities[index, 0] >= wake_threshold:
                predicted_labels.append(ThreeClassLabel.wake.value)
            else:
                if raw_performance.class_probabilities[index, 2] >= rem_threshold:
                    predicted_labels.append(ThreeClassLabel.rem.value)
                else:
                    predicted_labels.append(ThreeClassLabel.nrem.value)

        return np.array(predicted_labels)
