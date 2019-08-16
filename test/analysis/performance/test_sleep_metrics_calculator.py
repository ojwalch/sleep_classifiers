from unittest import TestCase
import numpy as np

from source.analysis.performance.sleep_metrics_calculator import SleepMetricsCalculator


class TestSleepMetricsCalculator(TestCase):

    def test_get_tst(self):
        labeled_night = np.array([0, 0, 1, 1, 1])
        tst = SleepMetricsCalculator.get_tst(labeled_night)

        self.assertEqual(1.5, tst)

    def test_get_waso(self):
        labeled_night = np.array([0, 0, 0, 0, 1, 2, 1, 0, 0, 0])
        waso = SleepMetricsCalculator.get_wake_after_sleep_onset(labeled_night)

        self.assertEqual(1.5, waso)

    def test_get_sleep_efficiency(self):
        labeled_night = np.array([0, 0, 1, 1, 2, 1, 2, 0, 1, 1])
        sleep_efficiency = SleepMetricsCalculator.get_sleep_efficiency(labeled_night)
        self.assertEqual(0.7, sleep_efficiency)

    def test_get_sleep_onset_latency(self):
        labeled_night = np.array([0, 0, 0, 1, 2, 2, 1, 0, 1, 1])
        sleep_onset_latency = SleepMetricsCalculator.get_sleep_onset_latency(labeled_night)
        self.assertEqual(1.5, sleep_onset_latency)

    def test_time_in_rem(self):
        labeled_night = np.array([0, 0, 0, 1, 2, 2, 1, 0, 1, 1])
        time_in_rem = SleepMetricsCalculator.get_time_in_rem(labeled_night)
        self.assertEqual(1, time_in_rem)

    def test_time_in_nrem(self):
        labeled_night = np.array([0, 0, 0, 1, 2, 2, 1, 0, 0, 1])
        time_in_nrem = SleepMetricsCalculator.get_time_in_nrem(labeled_night)
        self.assertEqual(1.5, time_in_nrem)
