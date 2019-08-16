from enum import Enum


class SleepWakeLabel(Enum):
    wake = 0
    sleep = 1


class ThreeClassLabel(Enum):
    wake = 0
    nrem = 1
    rem = 2
