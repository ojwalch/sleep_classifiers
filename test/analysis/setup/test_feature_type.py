from unittest import TestCase

from source.analysis.setup.feature_type import FeatureType


class TestFeatureType(TestCase):

    def test_types(self):
        self.assertEqual(FeatureType.count.value, "count")
        self.assertEqual(FeatureType.motion.value, "motion")
        self.assertEqual(FeatureType.heart_rate.value, "heart rate")
        self.assertEqual(FeatureType.cosine.value, "cosine")
        self.assertEqual(FeatureType.circadian_model.value, "circadian model")
        self.assertEqual(FeatureType.time.value, "time")
