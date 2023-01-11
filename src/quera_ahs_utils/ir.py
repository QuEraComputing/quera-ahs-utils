from braket.ahs.atom_arrangement import SiteType
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement

from typing import NoReturn,Tuple

# import simplejson as json
from braket.ir.ahs import Program

import json
import numpy as np


__all__ = [
    'to_json_file',
    'from_json_file',
    'quera_json_to_ahs',
    'braket_sdk_to_quera_json',
]

def to_json_file(js:dict,json_filename:str,**json_options) -> NoReturn:
    """prints out a dictionary to a json file. 

    Args:
        js (dict): data to be serialaized.
        json_filename (str): filename to output to.
        **json_options: options that get passed to the json serializer as `json.dump(...,**json_options)`
        
    """
    with open(json_filename,"w") as IO:
        json.dump(js,IO,**json_options)


def from_json_file(json_filename:str) -> dict:
    """deserialize a json file. 

    Args:
        json_filename (str): the json file to deserialize. 

    Returns:
        dict: the json file deserialized as a python dict. 
    """
    with open(json_filename,"w") as IO:
        js = json.load(IO)

    return js


def quera_json_to_ahs(js: dict) -> Tuple[int,AnalogHamiltonianSimulation]:
    """Convert a QuEra compatible program to a braket AHS program. 

    Args:
        js (dict): dictionary containing a program formatted to be accepted by 
        the QuEra API. 

    Returns:
        Tuple[int,AnalogHamiltonianSimulation]: A tuple continaing
            the number of shots as the first element and the ahs program 
            as the second argument. 
    """
    register = AtomArrangement()

    nshots = js["nshots"]

    lattice = js["lattice"]
    for coord,filling in zip(lattice["sites"],lattice["filling"]):
        site_type = SiteType.FILLED if filling==1 else SiteType.VACANT
        register.add(coord,site_type)

    rydberg = js["effective_hamiltonian"]["rydberg"]


    js_amplitude = rydberg["rabi_frequency_amplitude"]["global"]
    js_phase = rydberg["rabi_frequency_phase"]["global"]
    js_detuning = rydberg["detuning"]["global"]

    amplitude = TimeSeries()
    detuning = TimeSeries()  
    phase = TimeSeries()    

    for t,amplitude_value in zip(js_amplitude["times"],js_amplitude["values"]):
        amplitude.put(t, amplitude_value)

    for t,detuning_value in zip(js_detuning["times"],js_detuning["values"]):
        detuning.put(t, detuning_value)

    for t,phase_value in zip(js_phase["times"],js_phase["values"]):
        phase.put(t, phase_value) 


    drive = DrivingField(
        amplitude=amplitude, 
        detuning=detuning, 
        phase=phase
    )
        
    return nshots,AnalogHamiltonianSimulation(
            register=register, 
            hamiltonian=drive
        )


def braket_sdk_to_quera_json(ahs : AnalogHamiltonianSimulation, shots: int = 1) -> dict:
    """Translates Braket AHS IR program to Quera-compatible JSON.

    Args:
        ahs (AnalogHamiltonianSimulation): AHS object from braket SDK
        shots (int): The number of shots to run this program

    Returns:
        dict: Serialized QuEra-compatible dict representation of program
    """
    sites = []
    filling = []
    for site in ahs.register:
        x,y = site.coordinate
        site_type = site.site_type
        fill = 1 if site_type == SiteType.FILLED else 0
        sites.append([x,y])
        filling.append(fill)


    rabi_times = list(np.array(ahs.hamiltonian.amplitude.time_series.times(),dtype=np.float64))
    rabi_values = list(np.array(ahs.hamiltonian.amplitude.time_series.values(),dtype=np.float64))

    phase_times = list(np.array(ahs.hamiltonian.phase.time_series.times(),dtype=np.float64))
    phase_values = list(np.array(ahs.hamiltonian.phase.time_series.values(),dtype=np.float64))

    detuning_times = list(np.array(ahs.hamiltonian.detuning.time_series.times(),dtype=np.float64))
    detuning_values = list(np.array(ahs.hamiltonian.detuning.time_series.values(),dtype=np.float64))
    

    # QuEra IR Program
    translated_quera_program = dict()
    translated_quera_program["nshots"] = shots
    translated_quera_program["lattice"] = {"sites":sites,"filling":filling}
    translated_quera_program["effective_hamiltonian"] = {
        "rydberg": {
            "rabi_frequency_amplitude": {"global": {"times": rabi_times, "values": rabi_values}},
            "rabi_frequency_phase": {"global": {"times": phase_times, "values":phase_values}},
            "detuning": {
                "global": {"times": detuning_times, "values": detuning_values},
            },
        }
    }
    


    return translated_quera_program
