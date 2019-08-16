import csv

from source.preprocessing.epoch import Epoch
from source.preprocessing.psg.psg_converter import PSGConverter
from source.preprocessing.psg.stage_item import StageItem
from source.preprocessing.time_service import TimeService


class CompumedicsProcessor(object):
    DT_COMPUMEDICS_PSG = 30

    @staticmethod
    def parse(report_summary, psg_stage_path):
        data = []
        score_strings = []
        with open(psg_stage_path, 'rt') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

            for row in file_reader:
                score_strings.append(row[0])

        start_epoch = report_summary.start_epoch
        start_time_seconds = TimeService.get_start_epoch_timestamp(report_summary)

        for epoch_index in range(start_epoch - 1, len(score_strings)):
            timestamp = start_time_seconds + (epoch_index - start_epoch + 1) * CompumedicsProcessor.DT_COMPUMEDICS_PSG
            epoch = Epoch(timestamp=timestamp, index=epoch_index + 1)

            stage = PSGConverter.get_label_from_string(score_strings[epoch_index])
            data.append(StageItem(epoch=epoch, stage=stage))

        return data
