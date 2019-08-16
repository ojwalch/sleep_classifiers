from unittest import TestCase
import numpy as np

from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection
from source.preprocessing.interval import Interval


class TestHeartRateCollection(TestCase):
    def test_properties(self):
        subject_id = "subjectA"
        heart_rate_collection = HeartRateCollection(subject_id=subject_id,
                                                    data=np.array([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(subject_id, heart_rate_collection.subject_id)
        self.assertEqual(np.array([1, 4]).tolist(), heart_rate_collection.timestamps.tolist())
        self.assertEqual(np.array([[2, 3], [5, 6]]).tolist(), heart_rate_collection.values.tolist())

    def test_get_interval(self):
        heart_rate_collection = HeartRateCollection(subject_id="subjectA",
                                                    data=np.array([[1, 2, 3], [4, 5, 6]]))
        interval = Interval(start_time=1, end_time=4)
        self.assertEqual(interval.start_time, heart_rate_collection.get_interval().start_time)
        self.assertEqual(interval.end_time, heart_rate_collection.get_interval().end_time)
