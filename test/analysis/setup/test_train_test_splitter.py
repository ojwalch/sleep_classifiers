from unittest import TestCase

from source.analysis.setup.data_split import DataSplit
from source.analysis.setup.train_test_splitter import TrainTestSplitter
from test_helper import TestHelper


class TestTrainTestSplitter(TestCase):

    def test_leave_one_out(self):
        subject_ids = ["subjectA", "subjectB", "subjectC"]
        results = TrainTestSplitter.leave_one_out(subject_ids)

        TestHelper.assert_models_equal(self, DataSplit(training_set=["subjectB", "subjectC"], testing_set=["subjectA"]),
                                       results[0])
        TestHelper.assert_models_equal(self, DataSplit(training_set=["subjectA", "subjectC"], testing_set=["subjectB"]),
                                       results[1])
        TestHelper.assert_models_equal(self, DataSplit(training_set=["subjectA", "subjectB"], testing_set=["subjectC"]),
                                       results[2])

    def test_by_fraction(self):
        subject_ids = ["subjectA", "subjectB", "subjectC", "subjectD", "subjectE", "subjectF", "subjectG", "subjectH"]
        results = TrainTestSplitter.by_fraction(subject_ids=subject_ids, test_fraction=0.25, number_of_splits=5)

        self.assertEqual(5, len(results))
        training_set = results[3].training_set
        testing_set = results[3].testing_set

        self.assertEqual(len(training_set), 6)
        self.assertEqual(len(testing_set), 2)

        unique_elements = list(set(training_set) & set(testing_set))

        self.assertEqual(len(unique_elements), 0)
