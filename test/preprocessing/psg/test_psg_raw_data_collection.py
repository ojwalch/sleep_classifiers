from unittest import TestCase

from source.preprocessing.epoch import Epoch
from source.preprocessing.interval import Interval
from source.preprocessing.psg.psg_raw_data_collection import PSGRawDataCollection
from source.preprocessing.psg.stage_item import StageItem
from source.sleep_stage import SleepStage
from test.test_helper import TestHelper
import numpy as np


class TestPSGRawDataCollection(TestCase):

    def test_constructor_and_get_functions(self):
        subject_id = "subject9000"
        data = [StageItem(Epoch(timestamp=2, index=2), stage=SleepStage.rem),
                StageItem(Epoch(timestamp=200, index=4), stage=SleepStage.n1)]
        expected_interval = Interval(start_time=2, end_time=200)
        expected_np_array = np.array([[2, 5], [200, 1]])

        psg_raw_data_collection = PSGRawDataCollection(subject_id=subject_id, data=data)

        self.assertListEqual(data, psg_raw_data_collection.data)
        self.assertEqual(subject_id, psg_raw_data_collection.subject_id)

        TestHelper.assert_models_equal(self, expected_interval, psg_raw_data_collection.get_interval())
        self.assertListEqual(expected_np_array.tolist(), psg_raw_data_collection.get_np_array().tolist())
