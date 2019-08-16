from source.sleep_stage import SleepStage


class StageItem(object):
    def __init__(self, epoch, stage: SleepStage):
        self.epoch = epoch
        self.stage = stage
