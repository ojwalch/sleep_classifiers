from unittest import TestCase

from source.analysis.setup.data_split import DataSplit


class TestDataSplit(TestCase):
    def test_properties(self):
        training_set = ["3", "25", "35"]
        testing_set = ["2", "A32", "543"]
        split = DataSplit(training_set=training_set, testing_set=testing_set)

        self.assertListEqual(split.training_set, training_set)
        self.assertListEqual(split.testing_set, testing_set)
