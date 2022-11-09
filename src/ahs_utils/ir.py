from braket.ahs.atom_arrangement import SiteType
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement

from decimal import Decimal
from typing import Any, Dict

import simplejson as json
from braket.ir.ahs import Program

import json
import numpy


def to_json_file(js,json_filename,**json_options):
    with open(json_filename,"w") as IO:
        json.dump(js,IO,**json_options)



def from_json_file(json_filename):
    with open(json_filename,"w") as IO:
        js = json.load(IO)

    return js

def quera_json_to_ahs(js: dict) -> AnalogHamiltonianSimulation:
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

    for t,amplitude_value in zip(js_amplitude["times"],js_ampitude["values"]):
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

def braket_ir_to_quera_json(ahs_program: Program, shots: int = 1) -> dict:
    """Translates Braket AHS IR program to Quera-compatible JSON.

    Args:
        ahs_program (Program): AHS IR representation
        shots (int): The number of shots to run this program

    Returns:
        str: Serialized Quera-compatible JSON representation of program

    Raises:
        ValidationException: If any schema mismatch with Braket AHS and/or Quera task IR.
    """
    
    # AHS IR Program
    ahs_atom_array = ahs_program.setup.atomArray
    ahs_driving_fields = ahs_program.hamiltonian.drivingFields
    ahs_shifting_fields = ahs_program.hamiltonian.shiftingFields

    # QuEra IR Program
    translated_quera_program = dict()
    translated_quera_program["nshots"] = shots
    translated_quera_program["lattice"] = ahs_atom_array.dict()

    translated_quera_program["effective_hamiltonian"] = {
        "rydberg": {
            "rabi_frequency_amplitude": {"global": {"times": [], "values": []}},
            "rabi_frequency_phase": {"global": {"times": [], "values": []}},
            "detuning": {
                "global": {"times": [], "values": []},
            },
        }
    }
    rydberg_translated_quera_program = translated_quera_program["effective_hamiltonian"]["rydberg"]
    quera_rabi_frequency_amplitude = rydberg_translated_quera_program["rabi_frequency_amplitude"]["global"]
    quera_rabi_frequency_phase = rydberg_translated_quera_program["rabi_frequency_phase"]["global"]
    quera_detuning_global = rydberg_translated_quera_program["detuning"]["global"]

    for ahs_driving_field in ahs_driving_fields:
        quera_rabi_frequency_amplitude["values"].extend(ahs_driving_field.amplitude.sequence.values)
        quera_rabi_frequency_amplitude["times"].extend(ahs_driving_field.amplitude.sequence.times)

        quera_rabi_frequency_phase["values"].extend(ahs_driving_field.phase.sequence.values)
        quera_rabi_frequency_phase["times"].extend(ahs_driving_field.phase.sequence.times)

        quera_detuning_global["values"].extend(ahs_driving_field.detuning.sequence.values)
        quera_detuning_global["times"].extend(ahs_driving_field.detuning.sequence.times)


    if ahs_shifting_fields:
        quera_detuning_local = rydberg_translated_quera_program["detuning"]["local"] = []
        for ahs_shifting_field in ahs_shifting_fields:
            quera_detuning_local.append(
                {
                    "values": ahs_shifting_field.magnitude.sequence.values,
                    "times": ahs_shifting_field.magnitude.sequence.times,
                    "lattice_site_coefficients": ahs_shifting_field.magnitude.pattern,
                }
            )

    return json.loads(json.dumps(translated_quera_program,use_decimal=True))

def braket_sdk_to_quera_json(ahs : AnalogHamiltonianSimulation, shots: int = 1) -> dict:
    """Translates Braket AHS IR program to Quera-compatible JSON.

    Args:
        ahs (AnalogHamiltonianSimulation): AHS object from braket SDK
        shots (int): The number of shots to run this program

    Returns:
        str: Serialized Quera-compatible JSON representation of program

    Raises:
        ValidationException: If any schema mismatch with Braket AHS and/or Quera task IR.
    """
    return braket_ir_to_quera_json(ahs.to_ir(),shots)
