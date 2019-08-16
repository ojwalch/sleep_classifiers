from enum import Enum


class SleepStage(Enum):
    wake = 0
    n1 = 1
    n2 = 2
    n3 = 3
    n4 = 4
    rem = 5
    unscored = -1
