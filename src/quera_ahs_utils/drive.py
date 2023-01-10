import numpy as np

from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
from collections import Counter

from typing import Dict, List, Tuple
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result \
    import AnalogHamiltonianSimulationQuantumTaskResult
from braket.ahs.atom_arrangement import AtomArrangement


__all__ = [
    "rabi_pulse",
    "get_drive",
    "get_shift",
    "constant_time_series",
    "concatenate_time_series",
    "concatenate_drives",
    "concatenate_shifts",
    "concatenate_drive_list",
    "concatenate_shift_list"
]


def rabi_pulse(
    rabi_pulse_area: float, 
    omega_max: float,
    omega_slew_rate_max: float
) -> Tuple[List[float], List[float]]:
    """Get a time series for Rabi frequency with specified Rabi phase, maximum amplitude
        and maximum slew rate

        Args:
            rabi_pulse_area (float): Total area under the Rabi frequency time series
            omega_max (float): The maximum amplitude 
            omega_slew_rate_max (float): The maximum slew rate

        Returns:
            Tuple[List[float], List[float]]: A tuple containing the time points and values
                of the time series for the time dependent Rabi frequency

        Notes: By Rabi phase, it means the integral of the amplitude of a time-dependent 
            Rabi frequency, \int_0^T\Omega(t)dt, where T is the duration.
    """

    phase_threshold = omega_max**2 / omega_slew_rate_max
    if rabi_pulse_area <= phase_threshold:
        t_ramp = np.sqrt(rabi_pulse_area / omega_slew_rate_max)
        t_plateau = 0
    else:
        t_ramp = omega_max / omega_slew_rate_max
        t_plateau = (rabi_pulse_area / omega_max) - t_ramp
    t_pules = 2 * t_ramp + t_plateau
    time_points = [0, t_ramp, t_ramp + t_plateau, t_pules]
    amplitude_values = [0, t_ramp * omega_slew_rate_max, t_ramp * omega_slew_rate_max, 0]
    
    return time_points, amplitude_values


def get_drive(
    times: List[float], 
    amplitude_values: List[float], 
    detuning_values: List[float], 
    phase_values: List[float]
) -> DrivingField:
    """Get the driving field from a set of time points and values of the fields

        Args:
            times (List[float]): The time points of the driving field
            amplitude_values (List[float]): The values of the amplitude
            detuning_values (List[float]): The values of the detuning
            phase_values (List[float]): The values of the phase

        Returns:
            DrivingField: The driving field obtained
    """

    assert len(times) == len(amplitude_values)
    assert len(times) == len(detuning_values)
    assert len(times) == len(phase_values)
    
    amplitude = TimeSeries()
    detuning = TimeSeries()  
    phase = TimeSeries()    
    
    for t, amplitude_value, detuning_value, phase_value in zip(times, amplitude_values, detuning_values, phase_values):
        amplitude.put(t, amplitude_value)
        detuning.put(t, detuning_value)
        phase.put(t, phase_value) 

    drive = DrivingField(
        amplitude=amplitude, 
        detuning=detuning, 
        phase=phase
    )    
    
    return drive


def get_shift(times: List[float], values: List[float], pattern: List[float]) -> ShiftingField:
    """Get the shifting field from a set of time points, values and pattern

        Args:
            times (List[float]): The time points of the shifting field
            values (List[float]): The values of the shifting field
            pattern (List[float]): The pattern of the shifting field

        Returns:
            ShiftingField: The shifting field obtained
    """    
    assert len(times) == len(values)    
    
    magnitude = TimeSeries()
    for t, v in zip(times, values):
        magnitude.put(t, v)
    shift = ShiftingField(Field(magnitude, Pattern(pattern)))

    return shift


def constant_time_series(other_time_series: TimeSeries, constant: float=0.0) -> TimeSeries:
    """Obtain a constant time series with the same time points as the given time series

        Args:
            other_time_series (TimeSeries): The given time series

        Returns:
            TimeSeries: A constant time series with the same time points as the given time series
    """
    ts = TimeSeries()
    for t in other_time_series.times():
        ts.put(t, constant)
    return ts


def concatenate_time_series(time_series_1: TimeSeries, time_series_2: TimeSeries) -> TimeSeries:
    """Concatenate two time series to a single time series

        Args:
            time_series_1 (TimeSeries): The first time series to be concatenated
            time_series_2 (TimeSeries): The second time series to be concatenated

        Returns:
            TimeSeries: The concatenated time series

    """
    assert time_series_1.values()[-1] == time_series_2.values()[0]
    
    duration_1 = time_series_1.times()[-1] - time_series_1.times()[0]
    
    new_time_series = TimeSeries()
    new_times = time_series_1.times() + [t + duration_1 - time_series_2.times()[0] for t in time_series_2.times()[1:]]
    new_values = time_series_1.values() + time_series_2.values()[1:]
    for t, v in zip(new_times, new_values):
        new_time_series.put(t, v)
    
    return new_time_series


def concatenate_drives(drive_1: DrivingField, drive_2: DrivingField) -> DrivingField:
    """Concatenate two driving fields to a single driving field

        Args:
            drive_1 (DrivingField): The first driving field to be concatenated
            drive_2 (DrivingField): The second driving field to be concatenated

        Returns:
            DrivingField: The concatenated driving field
    """    
    return DrivingField(
        amplitude=concatenate_time_series(drive_1.amplitude.time_series, drive_2.amplitude.time_series),
        detuning=concatenate_time_series(drive_1.detuning.time_series, drive_2.detuning.time_series),
        phase=concatenate_time_series(drive_1.phase.time_series, drive_2.phase.time_series)
    )


def concatenate_shifts(shift_1: ShiftingField, shift_2: ShiftingField) -> ShiftingField:
    """Concatenate two driving fields to a single driving field

        Args:
            shift_1 (ShiftingField): The first shifting field to be concatenated
            shift_2 (ShiftingField): The second shifting field to be concatenated

        Returns:
            ShiftingField: The concatenated shifting field
    """        
    assert shift_1.magnitude.pattern.series == shift_2.magnitude.pattern.series
    
    new_magnitude = concatenate_time_series(shift_1.magnitude.time_series, shift_2.magnitude.time_series)
    return ShiftingField(Field(new_magnitude, shift_1.magnitude.pattern))


def concatenate_drive_list(drive_list: List[DrivingField]) -> DrivingField:
    """Concatenate a list of driving fields to a single driving field

        Args:
            drive_list (List[DrivingField]): The list of driving fields to be concatenated

        Returns:
            DrivingField: The concatenated driving field
    """        
    drive = drive_list[0]
    for dr in drive_list[1:]:
        drive = concatenate_drives(drive, dr)
    return drive    


def concatenate_shift_list(shift_list: List[ShiftingField]) -> ShiftingField:
    """Concatenate a list of shifting fields to a single driving field

        Args:
            shift_list (List[ShiftingField]): The list of shifting fields to be concatenated

        Returns:
            ShiftingField: The concatenated shifting field
    """            
    shift = shift_list[0]
    for sf in shift_list[1:]:
        shift = concatenate_shifts(shift, sf)
    return shift

