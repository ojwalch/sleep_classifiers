from unittest import TestCase

from source.analysis.performance.sleep_metrics import SleepMetrics


class TestSleepMetrics(TestCase):

    def test_properties(self):
        tst = 600
        sol = 45
        waso = 30
        sleep_efficiency = 0.3
        time_in_rem = 150
        time_in_nrem = 250
        sleep_metrics = SleepMetrics(tst=tst,
                                     sol=sol,
                                     waso=waso,
                                     sleep_efficiency=sleep_efficiency,
                                     time_in_rem=time_in_rem,
                                     time_in_nrem=time_in_nrem)

        self.assertEqual(tst, sleep_metrics.tst)
        self.assertEqual(sol, sleep_metrics.sol)
        self.assertEqual(waso, sleep_metrics.waso)
        self.assertEqual(sleep_efficiency, sleep_metrics.sleep_efficiency)
        self.assertEqual(time_in_rem, sleep_metrics.time_in_rem)
        self.assertEqual(time_in_nrem, sleep_metrics.time_in_nrem)
