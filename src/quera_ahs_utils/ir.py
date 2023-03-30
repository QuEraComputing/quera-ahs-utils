from braket.ahs.atom_arrangement import SiteType
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.pattern import Pattern
from braket.ahs.field import Field
import braket.ir.ahs as braket_ir

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement

from typing import NoReturn,Tuple, Optional

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


def get_register(lattice):
    register = AtomArrangement()
    for coord,filling in zip(lattice["sites"],lattice["filling"]):
        site_type = SiteType.FILLED if filling==1 else SiteType.VACANT
        register.add(coord, site_type)
    
    return register

def get_hamiltonian(effective_hamiltonian):
    rydberg = effective_hamiltonian["rydberg"]

    js_amplitude = rydberg["rabi_frequency_amplitude"]["global"]
    js_phase = rydberg["rabi_frequency_phase"]["global"]
    js_detuning = rydberg["detuning"]["global"]
    js_local_detuning = rydberg["detuning"].get("local", None)

    amplitude = TimeSeries()
    detuning = TimeSeries()  
    phase = TimeSeries()
    
    for t,amplitude_value in zip(js_amplitude["times"],js_amplitude["values"]):
        amplitude.put(t, amplitude_value)

    for t,detuning_value in zip(js_detuning["times"],js_detuning["values"]):
        detuning.put(t, detuning_value)

    for t,phase_value in zip(js_phase["times"],js_phase["values"]):
        phase.put(t, phase_value) 

    hamiltonian = DrivingField(
        amplitude=amplitude, 
        detuning=detuning, 
        phase=phase
    )

    if js_local_detuning != None:
        local_detuning = TimeSeries()
        for t,detuning_value in zip(js_local_detuning["times"],js_local_detuning["values"]):
            local_detuning.put(t, detuning_value) 

        hamiltonian = hamiltonian + ShiftingField(
            Field(
                    local_detuning, 
                    Pattern(js_local_detuning["lattice_site_coefficients"])
                )
            )
        
    return hamiltonian

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

    return js["nshots"],AnalogHamiltonianSimulation(
            register=get_register(js["lattice"]), 
            hamiltonian=get_hamiltonian(js["effective_hamiltonian"])
        )


def get_field(field: braket_ir.PhysicalField):
    times = np.array(field.time_series.times,dtype=np.float64)
    values = np.array(field.time_series.values,dtype=np.float64)
    
    times = np.around(times ,13)
    values = np.around(values, 13)
    
    return list(times), list(values), field.pattern
    
def get_local_detuning(shifting):
    local_times, local_values, lattice_site_coefficients = get_field(shifting.magnitude)
    
    if lattice_site_coefficients == 'uniform': 
        raise ValueError("local detuning must a list of detuning values, not 'uniform'")
    
    return {
                "lattice_site_coefficients": [float(coeff) for coeff in lattice_site_coefficients], 
                "times": local_times, 
                "values": local_values
            }
    
def get_detuning(driving, shifting = None):

    global_times, global_values, global_pattern = get_field(driving.detuning)

    if global_pattern != 'uniform': 
        raise ValueError("Detuning must have uniform pattern")
    
    if shifting is None:
        return {"global": {
                    "times": global_times, 
                    "values": global_values
                    }
                }
    else:
        return {"global": {
                    "times": global_times, 
                    "values": global_values
                    },
                "local": get_local_detuning(shifting)
                }
        
def get_rabi(driving):
    global_times, global_values, global_pattern = get_field(driving.amplitude)

    if global_pattern != 'uniform': 
        raise ValueError("Amplitude must have uniform pattern")

    return {"global": {"times": global_times, "values": global_values}}

def get_phase(driving):
    global_times, global_values, global_pattern = get_field(driving.phase)

    if global_pattern != 'uniform': 
        raise ValueError("Phase must have uniform pattern")

    return {"global": {"times": global_times, "values": global_values}}

def get_rydberg(driving, shifting = None):
    return {
        "rabi_frequency_amplitude": get_rabi(driving),
        "rabi_frequency_phase": get_phase(driving),
        "detuning": get_detuning(driving, shifting)
    }

def get_effective_hamiltonian(hamiltonian_ir):
    
    driving_fields = hamiltonian_ir.drivingFields
    shifting_fields = hamiltonian_ir.shiftingFields
    
    if len(driving_fields) != 1: raise ValueError("QuEra IR only supports exactly one set of driving fields")
    if len(shifting_fields) > 1:  raise ValueError("QuEra IR only supports at most one set of shifting fields")
    
    if len(shifting_fields) == 0:
        return {
            "rydberg": get_rydberg(driving_fields[0])
        }
    else:
        return {
            "rydberg": get_rydberg(driving_fields[0], shifting_fields[0])
        }       

def get_lattice(setup):
    sites = []
    for (x,y) in setup.ahs_register.sites:
        sites.append([float(x),float(y)])
        
    return {"sites":sites, "filling": setup.ahs_register.filling}

def braket_sdk_to_quera_json(ahs : AnalogHamiltonianSimulation, shots: int = 1) -> dict:
    """Translates Braket AHS IR program to Quera-compatible JSON.

    Args:
        ahs (AnalogHamiltonianSimulation): AHS object from braket SDK
        shots (int): The number of shots to run this program

    Returns:
        dict: Serialized QuEra-compatible dict representation of program
    """

    ahs_ir = ahs.to_ir()
    # QuEra IR Program
    return {
        "nshots": shots,
        "lattice": get_lattice(ahs_ir.setup),
        "effective_hamiltonian" : get_effective_hamiltonian(ahs_ir.hamiltonian)
    }
