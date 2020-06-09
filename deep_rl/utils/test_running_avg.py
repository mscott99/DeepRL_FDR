from unittest import TestCase
from . import RunningAvg, DualRunningAvg

class TestRunningAvg(TestCase):
    def test_get(self):
        run_avg_no_init = RunningAvg(5)
        run_avg_init_val = RunningAvg(5, False, 10)
        inputs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        for i in inputs:
            run_avg_no_init.add(i)
            run_avg_init_val.add(i)

        self.assertAlmostEqual(run_avg_no_init.get(), 1,places=7, msg="Failed equality test")
        self.assertNotAlmostEqual(run_avg_init_val.get(), 1,places=7, msg="initial bias malfunction")

class TestDualRunningAvg(TestCase):
    def test_get(self):
        run_avg_half_length = DualRunningAvg(3,1)
        run_avg_longer_mem = DualRunningAvg(3,2)
        single_elt = [1]
        many_elt = [1,1,1,1,1,1]
        many_diff_elts = [1,2,3,4,5]
        check_first_deleted = [0,0,0,0,0,1]
        for i in single_elt:
            run_avg_half_length.add(i)

        self.assertAlmostEqual(run_avg_half_length.get(),1.0)

        for i in many_elt:
            run_avg_half_length.add(i)
            run_avg_longer_mem.add(i)
        self.assertAlmostEqual(run_avg_half_length.get(), 1.0)
        self.assertAlmostEqual(run_avg_longer_mem.get(), 1.0)

        run_avg_half_length = DualRunningAvg(3, 1)
        run_avg_longer_mem = DualRunningAvg(3, 2)

        for i in many_diff_elts:
            run_avg_half_length.add(i)
            run_avg_longer_mem.add(i)

        self.assertGreater(run_avg_half_length.get(), run_avg_longer_mem.get())


        run_avg_half_length = DualRunningAvg(3, 1)
        run_avg_longer_mem = DualRunningAvg(3, 2)

        for i in check_first_deleted:
            run_avg_half_length.add(i)
            run_avg_longer_mem.add(i)

        #check averages have equal weight on the last element
        self.assertAlmostEqual(run_avg_half_length.get(), 0.0)
        self.assertAlmostEqual(run_avg_longer_mem.get(), 0.0)
