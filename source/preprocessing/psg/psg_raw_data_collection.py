import numpy as np

from source.preprocessing.interval import Interval
from source.sleep_stage import SleepStage


class PSGRawDataCollection(object):
    def __init__(self, subject_id, data: [SleepStage]):
        self.subject_id = subject_id
        self.data = data

    def get_np_array(self):
        number_of_epochs = len(self.data)
        array = np.zeros((number_of_epochs, 2))

        for index in range(number_of_epochs):
            stage_item = self.data[index]
            array[index, 0] = stage_item.epoch.timestamp
            array[index, 1] = stage_item.stage.value

        return array

    def get_interval(self):
        number_of_epochs = len(self.data)
        min_timestamp = 1e15
        max_timestamp = -1

        for index in range(number_of_epochs):
            stage_item = self.data[index]
            if stage_item.epoch.timestamp < min_timestamp:
                min_timestamp = stage_item.epoch.timestamp
            if stage_item.epoch.timestamp > max_timestamp:
                max_timestamp = stage_item.epoch.timestamp

        return Interval(start_time=min_timestamp, end_time=max_timestamp)
