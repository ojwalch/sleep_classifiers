from unittest import TestCase

from source.analysis.setup.sleep_label import SleepWakeLabel, ThreeClassLabel


class TestSleepLabel(TestCase):

    def test_label_enums(self):
        self.assertEqual(SleepWakeLabel.wake.value, 0)
        self.assertEqual(SleepWakeLabel.sleep.value, 1)
        self.assertEqual(ThreeClassLabel.wake.value, 0)
        self.assertEqual(ThreeClassLabel.nrem.value, 1)
        self.assertEqual(ThreeClassLabel.rem.value, 2)


