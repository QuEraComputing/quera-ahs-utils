import unittest
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField

from quera_ahs_utils.drive import slice_time_series,slice_drive,slice_shift
import numpy as np

class DriveModule(unittest.TestCase):
    @staticmethod
    def generate_time_series(times,values):
        time_series = TimeSeries()
        for time,value in zip(times,values):
            time_series.put(time,value)
        return time_series
    
    @staticmethod
    def time_series_all_close(first: TimeSeries, second: TimeSeries, reltol: float = 1.05e-8 ):
        if len(first) != len(second):
            return False
        if len(first) == 0:
            return True
        first_times = first.times()
        second_times = second.times()
        first_values = first.values()
        second_values = second.values()
        for index in range(len(first)):
            tolerance = max(abs(first_times[index]), abs(second_times[index])) * reltol
            if abs(first_times[index] - second_times[index]) >= tolerance:
                return False
            
            tolerance = max(abs(first_values[index]), abs(second_values[index])) * reltol
            if abs(first_values[index] - second_values[index]) >= tolerance:
                return False
            
        return True
    
    def test_slice_time_series_1(self):
        time_series =self.generate_time_series(
            [0.0, 0.5, 1.5, 2.0],
            [0.0, 1.0, 2.5, 0.0]
        )
        
        sliced_time_series = slice_time_series(time_series,0.25,1)
        test_time_series = self.generate_time_series(
            [0.0,0.25,0.75],
            [0.5,1.0,1.75]
        )
        self.assertEqual(sliced_time_series.times(),test_time_series.times())    
        self.assertEqual(sliced_time_series.values(),test_time_series.values())
        
        
    def test_slice_time_series_2(self):
        time_series =self.generate_time_series(
            [0.0, 0.5, 1.5, 2.0],
            [0.0, 1.0, 2.5, 0.0]
        )
        
        sliced_time_series = slice_time_series(time_series,0.0,2.0)
        self.assertEqual(sliced_time_series.times(),time_series.times())    
        self.assertEqual(sliced_time_series.values(),time_series.values())
        
    def test_slice_time_series_3(self):
        time_series =self.generate_time_series(
            [0.0, 0.5, 1.5, 2.0],
            [0.0, 1.0, 2.5, 0.0]
        )
        
        sliced_time_series = slice_time_series(time_series,0.0,1.0)
        test_time_series = self.generate_time_series(
            [0.0,0.5,1.0],
            [0.0,1.0,1.75]
        )
        
        self.assertEqual(sliced_time_series.times(),test_time_series.times())    
        self.assertEqual(sliced_time_series.values(),test_time_series.values())