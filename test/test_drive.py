import unittest
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField

import quera_ahs_utils.drive as drive
import numpy as np

class DriveModule(unittest.TestCase):
    
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
    
    def test_time_series(self):
        times = [0.0, 0.1, 0.5, 1.0]
        values = [0.0, 3.0, 2.3, 1.0]
        
        test_time_series = TimeSeries()
        test_time_series.put(0.0, 0.0)
        test_time_series.put(0.1, 3.0)
        test_time_series.put(0.5, 2.3)
        test_time_series.put(1.0, 1.0)
        
        generated_time_series = drive.time_series(times, values)
        
        self.assertEqual(generated_time_series.times(),test_time_series.times())    
        self.assertEqual(generated_time_series.values(),test_time_series.values())
        
    def test_get_time_sample_value_1(self):
        times = [0.0, 0.5, 2.0, 3.0]
        values = [0.0, 1.0, 3.0, 1.0]
        
        time_series = drive.time_series(times, values)

        samples = [
            drive.get_time_series_value(time_series, -0.5),
            drive.get_time_series_value(time_series,  0.25),
            drive.get_time_series_value(time_series,  2.0),
            drive.get_time_series_value(time_series, 4.0)
        ]
        self.assertEqual(samples,[0.0,0.5,3.0,1.0])
        
    def test_get_time_sample_value_2(self):
        times = [0.0, 0.5, 2.0, 3.0]
        values = [0.0, 1.0, 3.0, 1.0]
        
        time_series = drive.time_series(times, values)

        samples = [
            drive.get_time_series_value(time_series, -0.5, piecewise_constant=True),
            drive.get_time_series_value(time_series,  0.25, piecewise_constant=True),
            drive.get_time_series_value(time_series,  2.0, piecewise_constant=True),
            drive.get_time_series_value(time_series, 4.0, piecewise_constant=True)
        ]
        self.assertEqual(samples,[0.0,0.0,3.0,1.0])
        
    def test_slice_time_series_1(self):
        original_time_series = drive.time_series(
            [0.0, 0.5, 1.5, 2.0],
            [0.0, 1.0, 2.5, 0.0]
        )
        
        sliced_time_series = drive.slice_time_series(original_time_series,0.25,1)
        test_time_series = drive.time_series(
            [0.0,0.25,0.75],
            [0.5,1.0,1.75]
        )
        self.assertEqual(sliced_time_series.times(),test_time_series.times())    
        self.assertEqual(sliced_time_series.values(),test_time_series.values())
        
        
    def test_slice_time_series_2(self):
        original_time_series = drive.time_series(
            [0.0, 0.5, 1.5, 2.0],
            [0.0, 1.0, 2.5, 0.0]
        )
        
        sliced_time_series = drive.slice_time_series(original_time_series,0.0,2.0)
        self.assertEqual(sliced_time_series.times(),original_time_series.times())    
        self.assertEqual(sliced_time_series.values(),original_time_series.values())
        
    def test_slice_time_series_3(self):
        original_time_series = drive.time_series(
            [0.0, 0.5, 1.5, 2.0],
            [0.0, 1.0, 2.5, 0.0]
        )
        
        sliced_time_series = drive.slice_time_series(original_time_series,0.0,1.0)
        test_time_series = drive.time_series(
            [0.0,0.5,1.0],
            [0.0,1.0,1.75]
        )
        
        self.assertEqual(sliced_time_series.times(),test_time_series.times())    
        self.assertEqual(sliced_time_series.values(),test_time_series.values())
    

        