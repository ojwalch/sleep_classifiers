from source.sleep_stage import SleepStage


class PSGConverter(object):
    strings_to_labels = {
        "?": SleepStage.unscored,
        "W": SleepStage.wake,
        "1": SleepStage.n1,
        "N1": SleepStage.n1,
        "2": SleepStage.n2,
        "N2": SleepStage.n2,
        "3": SleepStage.n3,
        "N3": SleepStage.n3,
        "4": SleepStage.n4,
        "N4": SleepStage.n4,
        "R": SleepStage.rem,
        "M": SleepStage.wake}

    ints_to_labels = {
        -1: SleepStage.unscored,
        0: SleepStage.wake,
        1: SleepStage.n1,
        2: SleepStage.n2,
        3: SleepStage.n3,
        4: SleepStage.n4,
        5: SleepStage.rem,
        6: SleepStage.unscored}

    @staticmethod
    def get_label_from_string(stage_string):
        if stage_string in PSGConverter.strings_to_labels:
            return PSGConverter.strings_to_labels[stage_string]

    @staticmethod
    def get_label_from_int(stage_int):
        if stage_int in PSGConverter.ints_to_labels:
            return PSGConverter.ints_to_labels[stage_int]
