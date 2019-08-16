from unittest import TestCase

from source.sleep_stage import SleepStage


class TestSleepStage(TestCase):

    def test_stages(self):

        self.assertEqual(SleepStage.wake.value, 0)
        self.assertEqual(SleepStage.n1.value, 1)
        self.assertEqual(SleepStage.n2.value, 2)
        self.assertEqual(SleepStage.n3.value, 3)
        self.assertEqual(SleepStage.n4.value, 4)
        self.assertEqual(SleepStage.rem.value, 5)



