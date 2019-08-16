from unittest import TestCase

from source.preprocessing.psg.psg_converter import PSGConverter
from source.sleep_stage import SleepStage


class TestPSGConverter(TestCase):
    def test_converts_between_strings_and_labels(self):
        self.assertEqual(PSGConverter.get_label_from_string("W"), SleepStage.wake)
        self.assertEqual(PSGConverter.get_label_from_string("M"), SleepStage.wake)
        self.assertEqual(PSGConverter.get_label_from_string("1"), SleepStage.n1)
        self.assertEqual(PSGConverter.get_label_from_string("2"), SleepStage.n2)
        self.assertEqual(PSGConverter.get_label_from_string("3"), SleepStage.n3)
        self.assertEqual(PSGConverter.get_label_from_string("4"), SleepStage.n4)
        self.assertEqual(PSGConverter.get_label_from_string("N1"), SleepStage.n1)
        self.assertEqual(PSGConverter.get_label_from_string("N2"), SleepStage.n2)
        self.assertEqual(PSGConverter.get_label_from_string("N3"), SleepStage.n3)
        self.assertEqual(PSGConverter.get_label_from_string("R"), SleepStage.rem)
        self.assertEqual(PSGConverter.get_label_from_string("?"), SleepStage.unscored)

    def test_converts_between_ints_and_labels(self):
        self.assertEqual(PSGConverter.get_label_from_int(-1), SleepStage.unscored)
        self.assertEqual(PSGConverter.get_label_from_int(0), SleepStage.wake)
        self.assertEqual(PSGConverter.get_label_from_int(1), SleepStage.n1)
        self.assertEqual(PSGConverter.get_label_from_int(2), SleepStage.n2)
        self.assertEqual(PSGConverter.get_label_from_int(3), SleepStage.n3)
        self.assertEqual(PSGConverter.get_label_from_int(4), SleepStage.n4)
        self.assertEqual(PSGConverter.get_label_from_int(5), SleepStage.rem)
        self.assertEqual(PSGConverter.get_label_from_int(6), SleepStage.unscored)

