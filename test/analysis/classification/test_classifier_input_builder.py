from unittest import TestCase, mock
import numpy as np

from source.analysis.classification.classifier_input_builder import ClassifierInputBuilder
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject


class TestClassifierInputBuilder(TestCase):

    @mock.patch("source.analysis.classification.classifier_input_builder.ClassifierInputBuilder.get_array")
    def test_sleep_wake_inputs(self, mock_get_array):
        labels = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0])
        binarized_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
        values = np.array([3, 4])
        mock_get_array.return_value = values, labels
        subject_ids = []
        subject_dictionary = {}
        feature_set = []
        returned_values, returned_labels = ClassifierInputBuilder.get_sleep_wake_inputs(subject_ids,
                                                                                        subject_dictionary,
                                                                                        feature_set)
        self.assertListEqual(binarized_labels.tolist(), returned_labels.tolist())
        self.assertListEqual(values.tolist(), returned_values.tolist())

    def test_get_array(self):
        subject_ids = ["subjectA", "subjectB"]
        subject_dictionary = {
            "subjectA": Subject(subject_id="subjectA",
                                labeled_sleep=np.array([[0], [1]]),
                                feature_dictionary={FeatureType.count: np.array([[0], [1]]),
                                                    FeatureType.motion: np.array([[2], [3]]),
                                                    FeatureType.heart_rate: np.array([[4], [5]]),
                                                    FeatureType.cosine: np.array([[6], [7]]),
                                                    FeatureType.circadian_model: np.array([[8], [9]]),
                                                    FeatureType.time: np.array([[10], [11]]),
                                                    }),
            "subjectB": Subject(subject_id="subjectB",
                                labeled_sleep=np.array([[1], [1]]),
                                feature_dictionary={FeatureType.count: np.array([[100], [101]]),
                                                    FeatureType.motion: np.array([[102], [103]]),
                                                    FeatureType.heart_rate: np.array([[104], [105]]),
                                                    FeatureType.cosine: np.array([[106], [107]]),
                                                    FeatureType.circadian_model: np.array(
                                                        [[108], [109]]),
                                                    FeatureType.time: np.array([[110], [111]]),
                                                    })
        }

        feature_set = [FeatureType.count, FeatureType.cosine]

        features, labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)

        self.assertEqual(np.array([[0, 6], [1, 7], [100, 106], [101, 107]]).tolist(), features.tolist())
        self.assertEqual(np.array([[0], [1], [1], [1]]).tolist(), labels.tolist())
