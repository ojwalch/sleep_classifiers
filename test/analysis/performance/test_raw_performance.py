from unittest import TestCase
import numpy as np

from source.analysis.performance.raw_performance import RawPerformance


class TestRawPerformance(TestCase):

    def test_properties(self):

        true_labels = np.array([0, 1, 2])
        class_probabilities = np.array([[0.1, 0.9], [0, 1]])
        raw_performance = RawPerformance(true_labels=true_labels,
                                         class_probabilities=class_probabilities)
        self.assertEqual(raw_performance.true_labels.tolist(), true_labels.tolist())
        self.assertEqual(raw_performance.class_probabilities.tolist(), class_probabilities.tolist())
