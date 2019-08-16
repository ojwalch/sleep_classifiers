from unittest import TestCase
import numpy as np

from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject


class TestSubject(TestCase):

    def test_subject(self):
        subject_id = "subjectA"
        labeled_sleep = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        feature_dictionary = {FeatureType.count: np.array([]),
                              FeatureType.motion: np.array([]),
                              FeatureType.heart_rate: np.array([]),
                              FeatureType.cosine: np.array([]),
                              FeatureType.circadian_model: np.array([]),
                              FeatureType.time: np.array([])}
        subject = Subject(subject_id=subject_id,
                          labeled_sleep=labeled_sleep,
                          feature_dictionary=feature_dictionary)

        self.assertEqual(subject.subject_id, subject_id)
        self.assertListEqual(subject.labeled_sleep.tolist(), labeled_sleep.tolist())
        self.assertDictEqual(subject.feature_dictionary, feature_dictionary)
