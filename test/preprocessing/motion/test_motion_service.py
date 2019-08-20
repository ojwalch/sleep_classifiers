from unittest import TestCase, mock
import numpy as np

from source import utils
from source.constants import Constants
from source.preprocessing.interval import Interval
from source.preprocessing.motion.motion_collection import MotionCollection
from source.preprocessing.motion.motion_service import MotionService
from test.test_helper import TestHelper


class TestMotionService(TestCase):
    @mock.patch.object(utils, 'remove_repeats')
    @mock.patch.object(MotionService, 'load')
    @mock.patch.object(MotionService, 'get_raw_file_path')
    def test_load_raw(self, mock_get_raw_file_path, mock_load, mock_remove_repeats):
        subject_id = 'subjectA'
        mock_get_raw_file_path.return_value = returned_raw_path = 'path/to/file'
        mock_load.return_value = data = np.array([[1, 2, 3], [4, 5, 6]])
        mock_remove_repeats.return_value = data
        expected_motion_collection = MotionCollection(subject_id=subject_id, data=data)

        returned_motion_collection = MotionService.load_raw(subject_id)

        TestHelper.assert_models_equal(self, expected_motion_collection, returned_motion_collection)
        mock_get_raw_file_path.assert_called_once_with(subject_id)
        mock_load.assert_called_once_with(returned_raw_path)
        mock_remove_repeats.assert_called_once_with(data)

    @mock.patch.object(MotionService, 'load')
    @mock.patch.object(MotionService, 'get_cropped_file_path')
    def test_load_cropped(self, mock_get_cropped_file_path, mock_load):
        subject_id = 'subjectB'
        mock_get_cropped_file_path.return_value = returned_cropped_path = 'path/to/file'
        mock_load.return_value = data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected_motion_collection = MotionCollection(subject_id=subject_id, data=data)

        returned_motion_collection = MotionService.load_cropped(subject_id)

        TestHelper.assert_models_equal(self, expected_motion_collection, returned_motion_collection)
        mock_get_cropped_file_path.assert_called_once_with(subject_id)
        mock_load.assert_called_once_with(returned_cropped_path)

    @mock.patch('source.preprocessing.motion.motion_service.pd')
    def test_load(self, mock_pd):
        motion_file_path = 'path/to/file'
        mock_pd.read_csv.return_value.values = expected_data = np.array([1, 2, 3, 4, 5])

        returned_data = MotionService.load(motion_file_path)

        mock_pd.read_csv.assert_called_once_with(motion_file_path, delimiter=' ')
        self.assertEqual(expected_data.tolist(), returned_data.tolist())

    @mock.patch('source.preprocessing.motion.motion_service.np')
    @mock.patch.object(MotionService, 'get_cropped_file_path')
    def test_write(self, mock_cropped_file_path, mock_np):
        subject_id = 'subjectA'
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        motion_collection = MotionCollection(subject_id=subject_id, data=data)
        mock_cropped_file_path.return_value = file_path = 'path/to/file'

        MotionService.write(motion_collection)

        mock_np.savetxt.assert_called_once_with(file_path, motion_collection.data, fmt='%f')

    def test_crop(self):
        subject_id = 'subjectA'
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        motion_collection = MotionCollection(subject_id=subject_id, data=data)
        interval = Interval(start_time=4, end_time=9)
        cropped_data = np.array([[4, 5, 6], [7, 8, 9]])
        cropped_motion_collection = MotionCollection(subject_id=subject_id, data=cropped_data)

        returned_collection = MotionService.crop(motion_collection, interval)

        TestHelper.assert_models_equal(self, cropped_motion_collection, returned_collection)

    def test_get_cropped_file_path(self):
        subject_id = 'subject1'

        file_path = MotionService.get_cropped_file_path(subject_id)

        self.assertEqual(Constants.CROPPED_FILE_PATH.joinpath("subject1_cleaned_motion.out"), file_path)

    def test_get_raw_file_path(self):
        subject_id = 'subject1'
        motion_dir = utils.get_project_root().joinpath('data/motion/')

        file_path = MotionService.get_raw_file_path(subject_id)

        self.assertEqual(motion_dir.joinpath("subject1_acceleration.txt"), file_path)
