from unittest import TestCase

from source.analysis.setup.sleep_labeler import SleepLabeler
import numpy as np


class TestSleepLabeler(TestCase):

    def test_label_sleep_wake(self):
        sleep_wake_array = np.array([0, 1, 2, 3, 4, 5, 0, 4])
        labeled_sleep = SleepLabeler.label_sleep_wake(sleep_wake_array)

        self.assertEqual(len(sleep_wake_array), len(labeled_sleep))
        self.assertEqual(0, labeled_sleep[0])
        self.assertEqual(1, labeled_sleep[1])
        self.assertEqual(1, labeled_sleep[2])
        self.assertEqual(1, labeled_sleep[3])
        self.assertEqual(1, labeled_sleep[4])
        self.assertEqual(1, labeled_sleep[5])
        self.assertEqual(0, labeled_sleep[6])
        self.assertEqual(1, labeled_sleep[7])
