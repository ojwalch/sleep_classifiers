import csv

import numpy as np

from source.preprocessing.epoch import Epoch
from source.preprocessing.psg.psg_converter import PSGConverter
from source.preprocessing.psg.stage_item import StageItem
from source.preprocessing.time_service import TimeService


class VitaportProcessor(object):
    DT_TXT_PSG = 10

    @staticmethod
    def parse(report_summary, psg_stage_path):
        data = []
        with open(psg_stage_path, 'rt') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            count = 0
            rows_per_epoch = Epoch.DURATION / VitaportProcessor.DT_TXT_PSG

            for row in file_reader:
                if count == 0:
                    start_time = row[1]
                    start_score = int(row[0])
                    report_summary.start_time = start_time
                    start_time_seconds = TimeService.get_start_epoch_timestamp(report_summary)
                    epoch = Epoch(timestamp=start_time_seconds, index=1)
                    data.append(StageItem(epoch=epoch, stage=PSGConverter.get_label_from_int(start_score)))

                if np.mod(count, rows_per_epoch) == 0 and count != 0:
                    timestamp = start_time_seconds + count * VitaportProcessor.DT_TXT_PSG
                    score = int(row[0])
                    epoch = Epoch(timestamp=timestamp,
                                  index=(1 + int(np.floor(count / rows_per_epoch))))

                    data.append(StageItem(epoch=epoch, stage=PSGConverter.get_label_from_int(score)))
                count = count + 1

        return data
