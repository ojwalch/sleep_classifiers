from unittest import TestCase, mock

from source.mesa.mesa_psg_service import MesaPSGService
from source.mesa.mesa_subject_builder import MesaSubjectBuilder

import numpy as np

from source.preprocessing.activity_count.activity_count_collection import ActivityCountCollection
from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection


class TestMesaSubjectBuilder(TestCase):

    @mock.patch('source.mesa.mesa_subject_builder.MesaTimeBasedService')
    @mock.patch('source.mesa.mesa_subject_builder.MesaActigraphyService')
    @mock.patch('source.mesa.mesa_subject_builder.MesaHeartRateService')
    @mock.patch.object(MesaPSGService, 'load_raw')
    def test_build(self, mock_psg_service_load_raw, mock_heart_rate_service, mock_actigraphy_service,
                   mock_time_service):
        file_id = 'subjectA'

        returned_sleep_data = []
        for i in range(30):
            returned_sleep_data.append(0)
        for i in range(90):
            returned_sleep_data.append(1)
        for i in range(90):
            returned_sleep_data.append(3)
        for i in range(30):
            returned_sleep_data.append(2)

        heart_rate_data = np.array([[0, 20],
                                    [4, 60],
                                    [25, 95],
                                    [35, 333],
                                    [55, 25],
                                    [100, 233],
                                    [190, 24]])

        actigraphy_data = np.array([[0, 1],
                                    [15, 2],
                                    [30, 3],
                                    [45, 4],
                                    [60, 5],
                                    [75, 6],
                                    [90, 7]])

        circadian_data = np.array([[0, 0.3],
                                   [5, 0.4],
                                   [10, 0.1],
                                   [15, 0.6],
                                   [20, 0.7],
                                   [25, 0.2],
                                   [30, 0.1],
                                   [35, 0.3],
                                   [40, 0.4],
                                   [45, 0.1],
                                   [50, 0.6],
                                   [55, 0.7],
                                   [60, 0.2],
                                   [65, 0.1]])

        mock_psg_service_load_raw.return_value = np.array(returned_sleep_data)
        mock_heart_rate_service.load_raw.return_value = HeartRateCollection(subject_id=file_id, data=heart_rate_data)
        mock_actigraphy_service.load_raw.return_value = ActivityCountCollection(subject_id=file_id,
                                                                                data=actigraphy_data)
        mock_time_service.load_circadian_model.return_value = circadian_data

        subject = MesaSubjectBuilder.build(file_id)

        self.assertEqual([[0], [1], [1], [1], [3], [3], [3], [2]], subject.labeled_sleep.tolist())
