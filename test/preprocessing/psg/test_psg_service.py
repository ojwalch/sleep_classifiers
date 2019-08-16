from unittest import TestCase, mock

from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.interval import Interval
from source.preprocessing.psg.psg_raw_data_collection import PSGRawDataCollection
from source.preprocessing.psg.psg_service import PSGService
import numpy as np

from source.preprocessing.psg.stage_item import StageItem
from source.sleep_stage import SleepStage
from test.test_helper import TestHelper


class TestPSGService(TestCase):

    def test_crop(self):
        subject_id = 'subjectA'
        crop_interval = Interval(start_time=50, end_time=101)
        data = [StageItem(epoch=Epoch(timestamp=10, index=2), stage=SleepStage.n1),
                StageItem(epoch=Epoch(timestamp=50, index=3), stage=SleepStage.n2),
                StageItem(epoch=Epoch(timestamp=100, index=4), stage=SleepStage.n3),
                StageItem(epoch=Epoch(timestamp=170, index=5), stage=SleepStage.rem)]
        expected_data = [StageItem(epoch=Epoch(timestamp=50, index=3), stage=SleepStage.n2),
                         StageItem(epoch=Epoch(timestamp=100, index=4), stage=SleepStage.n3), ]

        input_raw_data_collection = PSGRawDataCollection(subject_id=subject_id, data=data)

        returned_psg_raw_collection = PSGService.crop(input_raw_data_collection, crop_interval)

        self.assertEqual(subject_id, returned_psg_raw_collection.subject_id)
        TestHelper.assert_models_equal(self, expected_data[0], returned_psg_raw_collection.data[0])
        TestHelper.assert_models_equal(self, expected_data[1], returned_psg_raw_collection.data[1])

    @mock.patch('source.preprocessing.psg.psg_service.np')
    def test_write(self, mock_np):
        subject_id = 'subjectA'
        mock_np.array.return_value = returned_array = np.array([[1, 2], [3, 4]])
        list_data = [[10, 1], [50, 2], [100, 3], [170, 5]]
        data = [StageItem(epoch=Epoch(timestamp=10, index=2), stage=SleepStage.n1),
                StageItem(epoch=Epoch(timestamp=50, index=3), stage=SleepStage.n2),
                StageItem(epoch=Epoch(timestamp=100, index=4), stage=SleepStage.n3),
                StageItem(epoch=Epoch(timestamp=170, index=5), stage=SleepStage.rem)]
        psg_raw_data_collection = PSGRawDataCollection(subject_id=subject_id, data=data)
        psg_output_path = Constants.CROPPED_FILE_PATH.joinpath("subjectA_cleaned_psg.out")

        PSGService.write(psg_raw_data_collection)

        mock_np.array.assert_called_once_with(list_data)
        mock_np.savetxt.assert_called_once_with(psg_output_path, returned_array, fmt='%f')

    @mock.patch('source.preprocessing.psg.psg_service.pd')
    def test_load_cropped_array(self, mock_pd):
        subject_id = 'subject100'
        cropped_psg_path = Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_psg.out")
        mock_pd.read_csv.return_value.values = np.array([[1, 2], [4, 5], [10, 1]])

        psg_collection = PSGService.load_cropped(subject_id)

        mock_pd.read_csv.assert_called_once_with(str(cropped_psg_path), delimiter=' ')
        TestHelper.assert_models_equal(self, StageItem(epoch=Epoch(timestamp=1, index=0), stage=SleepStage.n2),
                                       psg_collection.data[0])
        TestHelper.assert_models_equal(self, StageItem(epoch=Epoch(timestamp=4, index=1), stage=SleepStage.rem),
                                       psg_collection.data[1])
        TestHelper.assert_models_equal(self, StageItem(epoch=Epoch(timestamp=10, index=2), stage=SleepStage.n1),
                                       psg_collection.data[2])
