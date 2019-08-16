from unittest import TestCase
import numpy as np

from source.preprocessing.motion.motion_collection import MotionCollection
from source.preprocessing.interval import Interval


class TestMotionCollection(TestCase):
    def test_properties(self):
        subject_id = "subjectA"
        motion_collection = MotionCollection(subject_id=subject_id,
                                             data=np.array([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(subject_id, motion_collection.subject_id)
        self.assertEqual(np.array([1, 4]).tolist(), motion_collection.timestamps.tolist())
        self.assertEqual(np.array([[2, 3], [5, 6]]).tolist(), motion_collection.values.tolist())

    def test_get_interval(self):
        motion_collection = MotionCollection(subject_id="subjectA",
                                                 data=np.array([[1, 2, 3], [4, 5, 6]]))
        interval = Interval(start_time=1, end_time=4)
        self.assertEqual(interval.start_time, motion_collection.get_interval().start_time)
        self.assertEqual(interval.end_time, motion_collection.get_interval().end_time)
