from enum import Enum


class FeatureType(Enum):
    count = "count"
    motion = "motion"
    heart_rate = "heart rate"
    cosine = "cosine"
    circadian_model = "circadian model"
    time = "time"
