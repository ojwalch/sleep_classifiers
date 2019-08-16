from unittest import TestCase
import numpy as np
from sklearn.metrics import precision_score, auc, roc_curve, cohen_kappa_score

from source.analysis.performance.performance_builder import PerformanceBuilder
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.performance.epoch_performance import SleepWakePerformance
from test.test_helper import TestHelper


class TestPerformanceBuilder(TestCase):

    def test_build_from_raw(self):
        threshold = 0.2
        raw_performance = RawPerformance(true_labels=np.array([1, 0]),
                                         class_probabilities=np.array([[0.1, 0.9], [0.3, 0.7]]))

        predicted_labels = np.array([1, 1])
        kappa = cohen_kappa_score(raw_performance.true_labels, predicted_labels)

        sleep_predictive_value = precision_score(raw_performance.true_labels, predicted_labels, pos_label=1)
        wake_predictive_value = precision_score(raw_performance.true_labels, predicted_labels, pos_label=0)
        false_positive_rates, true_positive_rates, thresholds = roc_curve(raw_performance.true_labels,
                                                                          raw_performance.class_probabilities[:, 1],
                                                                          pos_label=1,
                                                                          drop_intermediate=False)
        auc_value = auc(false_positive_rates, true_positive_rates)

        expected_performance = SleepWakePerformance(accuracy=np.float64(0.5), wake_correct=np.float64(0),
                                                    sleep_correct=np.float64(1.0), kappa=kappa,
                                                    auc=auc_value, sleep_predictive_value=sleep_predictive_value,
                                                    wake_predictive_value=wake_predictive_value)

        performance = PerformanceBuilder.build_with_sleep_threshold(raw_performance, threshold)
        TestHelper.assert_models_equal(self, expected_performance, performance)
