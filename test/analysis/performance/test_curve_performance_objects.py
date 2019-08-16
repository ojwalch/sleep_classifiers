from unittest import TestCase

import numpy as np

from source.analysis.performance.curve_performance import ROCPerformance, PrecisionRecallPerformance


class TestROCPerformance(TestCase):

    def test_properties(self):
        true_positive_rates = np.array([1, 2])
        false_positive_rates = np.array([3, 4])
        roc_performance = ROCPerformance(true_positive_rates=true_positive_rates,
                                         false_positive_rates=false_positive_rates)

        self.assertListEqual(true_positive_rates.tolist(), roc_performance.true_positive_rates.tolist())
        self.assertListEqual(false_positive_rates.tolist(), roc_performance.false_positive_rates.tolist())


class TestPRPerformance(TestCase):

    def test_properties(self):
        precisions = np.array([1, 2])
        recalls = np.array([3, 4])
        precision_recall_performance = PrecisionRecallPerformance(precisions=precisions,
                                                                  recalls=recalls)

        self.assertListEqual(precisions.tolist(), precision_recall_performance.precisions.tolist())
        self.assertListEqual(recalls.tolist(), precision_recall_performance.recalls.tolist())
