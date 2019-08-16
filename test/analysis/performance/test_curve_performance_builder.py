from unittest import TestCase

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.performance.curve_performance_builder import CurvePerformanceBuilder
from source.analysis.setup.sleep_label import SleepWakeLabel


class TestCurvePerformanceBuilder(TestCase):

    def test_get_axes_bins(self):
        horizontal_axis, vertical_axis = CurvePerformanceBuilder.get_axes_bins()
        self.assertEqual(100, np.shape(horizontal_axis)[0])
        self.assertEqual(100, np.shape(vertical_axis)[0])
        self.assertEqual(0.01, horizontal_axis[1] - horizontal_axis[0])
        self.assertEqual(0, vertical_axis[0])

    def test_build_roc_from_raw_performances(self):
        raw_performances = [RawPerformance(true_labels=np.array([0, 1]),
                                           class_probabilities=np.array([[0, 1], [1, 0]])),
                            RawPerformance(true_labels=np.array([0, 1]),
                                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        first_false_positive_rates, first_true_positive_rates, first_thresholds = roc_curve(
            raw_performances[0].true_labels,
            raw_performances[0].class_probabilities[:, 1],
            pos_label=SleepWakeLabel.sleep.value,
            drop_intermediate=False)

        second_false_positive_rates, second_true_positive_rates, second_thresholds = roc_curve(
            raw_performances[1].true_labels,
            raw_performances[1].class_probabilities[:, 1],
            pos_label=SleepWakeLabel.sleep.value,
            drop_intermediate=False)

        horizontal_axis_bins, vertical_axis_bins = CurvePerformanceBuilder.get_axes_bins()

        first_interpolated_true_positive_rates = np.interp(horizontal_axis_bins, first_false_positive_rates,
                                                           first_true_positive_rates)

        second_interpolated_true_positive_rates = np.interp(horizontal_axis_bins, second_false_positive_rates,
                                                            second_true_positive_rates)

        expected_true_positive_rates = (first_interpolated_true_positive_rates +
                                        second_interpolated_true_positive_rates) / 2

        horizontal_axis_bins = np.insert(horizontal_axis_bins, 0, 0, axis=0)
        expected_true_positive_rates = np.insert(expected_true_positive_rates, 0, 0, axis=0)

        roc_performance = CurvePerformanceBuilder.build_roc_from_raw(raw_performances, 1)

        self.assertListEqual(horizontal_axis_bins.tolist(), roc_performance.false_positive_rates.tolist())
        self.assertListEqual(expected_true_positive_rates.tolist(), roc_performance.true_positive_rates.tolist())

    def test_build_pr_from_raw_performances(self):
        raw_performances = [RawPerformance(true_labels=np.array([0, 1]),
                                           class_probabilities=np.array([[0, 1], [1, 0]])),
                            RawPerformance(true_labels=np.array([0, 1]),
                                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        first_precisions, first_recalls, first_thresholds = precision_recall_curve(
            raw_performances[0].true_labels,
            raw_performances[0].class_probabilities[:, 0],
            pos_label=SleepWakeLabel.wake.value)

        second_precisions, second_recalls, second_thresholds = precision_recall_curve(
            raw_performances[1].true_labels,
            raw_performances[1].class_probabilities[:, 0],
            pos_label=SleepWakeLabel.wake.value)

        horizontal_axis_bins, vertical_axis_bins = CurvePerformanceBuilder.get_axes_bins()

        first_interpolated_precisions = np.interp(horizontal_axis_bins, np.flip(first_recalls),
                                                  np.flip(first_precisions))

        second_interpolated_precisions = np.interp(horizontal_axis_bins, np.flip(second_recalls),
                                                   np.flip(second_precisions))

        expected_precisions = (first_interpolated_precisions +
                               second_interpolated_precisions) / 2

        horizontal_axis_bins = np.insert(horizontal_axis_bins, 0, 0, axis=0)
        expected_precisions = np.insert(expected_precisions, 0, 1, axis=0)

        pr_performance = CurvePerformanceBuilder.build_precision_recall_from_raw(raw_performances)

        self.assertListEqual(horizontal_axis_bins.tolist(), pr_performance.recalls.tolist())
        self.assertListEqual(expected_precisions.tolist(), pr_performance.precisions.tolist())
