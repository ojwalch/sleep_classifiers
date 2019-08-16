from unittest import TestCase

import numpy as np
from source.preprocessing.activity_count.activity_count_collection import ActivityCountCollection
from source.preprocessing.interval import Interval


class TestActivityCountCollection(TestCase):

    def test_properties(self):
        subject_id = "subjectA"
        activity_count_collection = ActivityCountCollection(subject_id=subject_id,
                                                            data=np.array([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(subject_id, activity_count_collection.subject_id)
        self.assertEqual(np.array([1, 4]).tolist(), activity_count_collection.timestamps.tolist())
        self.assertEqual(np.array([[2, 3], [5, 6]]).tolist(), activity_count_collection.values.tolist())

    def test_get_interval(self):
        activity_count_collection = ActivityCountCollection(subject_id="subjectA",
                                                            data=np.array([[1, 2, 3], [4, 5, 6]]))
        interval = Interval(start_time=1, end_time=4)
        self.assertEqual(interval.start_time, activity_count_collection.get_interval().start_time)
        self.assertEqual(interval.end_time, activity_count_collection.get_interval().end_time)
