class SleepMetrics(object):
    def __init__(self, tst, sol, waso, sleep_efficiency, time_in_rem, time_in_nrem):
        self.tst = tst
        self.sol = sol
        self.waso = waso
        self.sleep_efficiency = sleep_efficiency
        self.time_in_rem = time_in_rem
        self.time_in_nrem = time_in_nrem