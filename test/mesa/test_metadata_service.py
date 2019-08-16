from unittest import TestCase, mock

from source.mesa.metadata_service import MetadataService
from mock import mock_open


class TestMetadataService(TestCase):

    @mock.patch('source.mesa.metadata_service.csv')
    @mock.patch("builtins.open", new_callable=mock_open, read_data='')
    def test_gets_metadata_dictionary(self, mock_open, mock_csv):
        subject_id = 1234567
        first_row = ['subject_id',
                     'ahiu35',
                     'sleepage5c',
                     'gender1',
                     'slpprdp5',
                     'time_bed5',
                     'waso5',
                     'slp_eff5',
                     'timerem5',
                     'timest15',
                     'timest25',
                     'timest345']

        second_row = [str(subject_id), '0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

        first_subject_dictionary = {'ahi': 0,
                                    'age': 10,
                                    'gender': 20,
                                    'tst': 30,
                                    'tib': 40,
                                    'waso': 50,
                                    'slp_eff': 60,
                                    'time_rem': 70,
                                    'time_n1': 80,
                                    'time_n2': 90,
                                    'time_n34': 100}

        mock_csv.reader.return_value = [first_row,
                                        second_row]

        dictionary = MetadataService.get_metadata_dictionary()

        self.assertDictEqual({subject_id: first_subject_dictionary}, dictionary)
