from unittest import TestCase, mock

from source import utils
from source.constants import Constants
from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
import numpy as np

from source.preprocessing.interval import Interval
from test.test_helper import TestHelper


class TestHeartRateService(TestCase):

    @mock.patch.object(utils, 'remove_repeats')
    @mock.patch.object(HeartRateService, 'load')
    @mock.patch.object(HeartRateService, 'get_raw_file_path')
    def test_load_raw(self, mock_get_raw_file_path, mock_load, mock_remove_repeats):
        subject_id = 'subjectA'
        mock_get_raw_file_path.return_value = returned_raw_path = 'path/to/file'
        mock_load.return_value = data = np.array([[1, 2, 3], [4, 5, 6]])
        mock_remove_repeats.return_value = data
        expected_heart_rate_collection = HeartRateCollection(subject_id=subject_id, data=data)

        returned_heart_rate_collection = HeartRateService.load_raw(subject_id)

        TestHelper.assert_models_equal(self, expected_heart_rate_collection, returned_heart_rate_collection)
        mock_get_raw_file_path.assert_called_once_with(subject_id)
        mock_load.assert_called_once_with(returned_raw_path, ',')
        mock_remove_repeats.assert_called_once_with(data)

    @mock.patch.object(HeartRateService, 'load')
    @mock.patch.object(HeartRateService, 'get_cropped_file_path')
    def test_load_cropped(self, mock_get_cropped_file_path, mock_load):
        subject_id = 'subjectB'
        mock_get_cropped_file_path.return_value = returned_cropped_path = 'path/to/file'
        mock_load.return_value = data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected_heart_rate_collection = HeartRateCollection(subject_id=subject_id, data=data)

        returned_heart_rate_collection = HeartRateService.load_cropped(subject_id)

        TestHelper.assert_models_equal(self, expected_heart_rate_collection, returned_heart_rate_collection)
        mock_get_cropped_file_path.assert_called_once_with(subject_id)
        mock_load.assert_called_once_with(returned_cropped_path)

    @mock.patch('source.preprocessing.heart_rate.heart_rate_service.pd')
    def test_load(self, mock_pd):
        heart_rate_file_path = 'path/to/file'
        mock_pd.read_csv.return_value.values = expected_data = np.array([1, 2, 3, 4, 5])

        returned_data = HeartRateService.load(heart_rate_file_path)

        mock_pd.read_csv.assert_called_once_with(heart_rate_file_path, delimiter=' ')
        self.assertEqual(expected_data.tolist(), returned_data.tolist())

    @mock.patch('source.preprocessing.heart_rate.heart_rate_service.np')
    @mock.patch.object(HeartRateService, 'get_cropped_file_path')
    def test_write(self, mock_cropped_file_path, mock_np):
        subject_id = 'subjectA'
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        heart_rate_collection = HeartRateCollection(subject_id=subject_id, data=data)
        mock_cropped_file_path.return_value = file_path = 'path/to/file'

        HeartRateService.write(heart_rate_collection)

        mock_np.savetxt.assert_called_once_with(file_path, heart_rate_collection.data, fmt='%f')

    def test_crop(self):
        subject_id = 'subjectA'
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        heart_rate_collection = HeartRateCollection(subject_id=subject_id, data=data)
        interval = Interval(start_time=4, end_time=9)
        cropped_data = np.array([[4, 5, 6], [7, 8, 9]])
        cropped_heart_rate_collection = HeartRateCollection(subject_id=subject_id, data=cropped_data)

        returned_collection = HeartRateService.crop(heart_rate_collection, interval)

        TestHelper.assert_models_equal(self, cropped_heart_rate_collection, returned_collection)

    def test_get_cropped_file_path(self):
        subject_id = 'subject1'

        file_path = HeartRateService.get_cropped_file_path(subject_id)

        self.assertEqual(Constants.CROPPED_FILE_PATH.joinpath("subject1_cleaned_hr.out"), file_path)

    def test_get_raw_file_path(self):
        subject_id = 'subject1'
        heart_rate_dir = utils.get_project_root().joinpath('data/heart_rate/')

        file_path = HeartRateService.get_raw_file_path(subject_id)

        self.assertEqual(heart_rate_dir.joinpath("subject1_heartrate.txt"), file_path)
