from unittest import TestCase, mock

from mock import mock_open, call
import numpy as np
from source.mesa.mesa_actigraphy_service import MesaActigraphyService


class TestMesaActigraphyService(TestCase):

    @mock.patch('source.mesa.mesa_actigraphy_service.np')
    @mock.patch('source.mesa.mesa_actigraphy_service.csv')
    @mock.patch("builtins.open", new_callable=mock_open, read_data='')
    @mock.patch('source.mesa.mesa_actigraphy_service.utils')
    def test_load_raw(self, mock_utils, mock_open, mock_csv, mock_np):
        file_id = '3'
        align_line_1 = ['1', '244', '20:20:00', '20:21:30']
        align_line_2 = ['3', '104', '14:20:00', '14:21:30']

        sleep_row_1 = ['3', '103', '20:50:00', '0', '0.00', '0', '2.7500', '0.5790', '0.1650', '0.0206', '0', 'ACTIVE',
                       '5', '1', '1']
        sleep_row_2 = ['3', '104', '20:50:30', '0', '0.00', '0', '2.7500', '0.5790', '0.1650', '0.0206', '0', 'ACTIVE',
                       '5', '1', '1']
        sleep_row_3 = ['3', '105', '20:51:00', '0', '200.0', '0', '2.7500', '0.5790', '0.1650', '0.0206', '0',
                       'ACTIVE', '5', '1', '1']

        expected_data = [[0.0, 0.0], [30.0, 200.0]]

        mock_csv.reader.side_effect = [[align_line_1,
                                        align_line_2],
                                       [sleep_row_1,
                                        sleep_row_2,
                                        sleep_row_3]]

        mock_np.array.return_value = array_return_value = 'value_returned_from_array'
        mock_utils.remove_nans.return_value = remove_nan_return_value = np.array([[0, 1], [2, 3]])

        activity_count_collection = MesaActigraphyService.load_raw(file_id)

        mock_utils.get_project_root.assert_called_once()
        mock_np.array.assert_called_once_with(expected_data)
        mock_utils.remove_nans.assert_called_once_with(array_return_value)

        self.assertEqual(remove_nan_return_value.tolist(), activity_count_collection.data.tolist())
