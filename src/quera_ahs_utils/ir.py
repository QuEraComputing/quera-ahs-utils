from __future__ import annotations

from braket.ahs.atom_arrangement import SiteType
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.pattern import Pattern
from braket.ahs.field import Field
import braket.ir.ahs as braket_ir

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement

from typing import NoReturn, Optional, Tuple, Union

# import simplejson as json
from braket.ir.ahs import Program

import json
import numpy as np

import quera_ahs_utils.drive as drive
from quera_ahs_utils.quera_ir.task_specification import (QuEraTaskSpecification, 
    Lattice, EffectiveHamiltonian, RydbergHamiltonian, RabiFrequencyAmplitude,
    RabiFrequencyPhase, Detuning, GlobalField, LocalField)



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


class quera_to_braket:

    @staticmethod
    def get_register(lattice: Lattice):
        register = AtomArrangement()
        for coord,filling in zip(lattice.sites, lattice.filling):
            site_type = SiteType.FILLED if filling==1 else SiteType.VACANT
            register.add(coord, site_type)
        
        return register

    @staticmethod
    def get_braket_field(quera_field: Optional[Union[GlobalField, LocalField]]):
        if isinstance(quera_field, GlobalField):
            pattern = Pattern("uniform")
            time_series = drive.time_series(quera_field.times, quera_field.values)
        elif isinstance(quera_field, LocalField):
            pattern = Pattern(quera_field.lattice_site_coefficients)
            time_series = drive.time_series(quera_field.times, quera_field.values)
        elif quera_field == None:
            return None
        else:
            raise TypeError("expecting quera_field to be one of quera_ir.GlobalField, quera_ir.LocalField.")
        
        return Field(time_series=time_series, pattern=pattern)    

    @staticmethod
    def get_amplitude(amplitude: RabiFrequencyAmplitude):
        return quera_to_braket.get_braket_field(amplitude.global_)

    @staticmethod
    def get_phase(phase: RabiFrequencyPhase):
        return quera_to_braket.get_braket_field(phase.global_)

    @staticmethod
    def get_detuning(detuning: Detuning):
        return quera_to_braket.get_braket_field(detuning.global_)

    @staticmethod
    def get_local_shifting_field(detuning: Detuning):
        return quera_to_braket.get_braket_field(detuning.local)
    
    @staticmethod
    def get_driving_field(rydberg: RydbergHamiltonian):
        return DrivingField(
            amplitude=quera_to_braket.get_amplitude(rydberg.rabi_frequency_amplitude), 
            detuning=quera_to_braket.get_detuning(rydberg.detuning), 
            phase=quera_to_braket.get_phase(rydberg.rabi_frequency_phase)
        )
    
    @staticmethod
    def get_shifting_field(rydberg: RydbergHamiltonian):
        magnitude = quera_to_braket.get_braket_field(rydberg.detuning.local)
        
        if magnitude != None:        
            return ShiftingField(magnitude)
        else:
            return None

    @staticmethod 
    def get_hamiltonian(effective_hamiltonian: EffectiveHamiltonian ):
        driving_field = quera_to_braket.get_driving_field(effective_hamiltonian.rydberg)
        shifting_field = quera_to_braket.get_shifting_field(effective_hamiltonian.rydberg)

        if shifting_field != None:
            return driving_field + shifting_field
        else:
            return driving_field
            
class braket_to_quera:

    @staticmethod
    def get_global_field(field: braket_ir.PhysicalField) -> GlobalField:
        if not isinstance(field, braket_ir.PhysicalField): 
            raise ValueError("expecting PhysicalField")
        
        if field.pattern != "uniform": 
            raise ValueError("global field must have 'uniform' pattern.")

        return GlobalField(
                times=field.time_series.times,
                values=field.time_series.values
            )
        
    @staticmethod
    def get_local_field(field: braket_ir.PhysicalField) -> LocalField:
        if not isinstance(field, braket_ir.PhysicalField): 
            raise ValueError("expecting PhysicalField")
        
        if field.pattern == "uniform": 
            raise ValueError("local field must have a list of real values for the pattern.")

        return LocalField(
                times=field.time_series.times,
                values=field.time_series.values,
                lattice_site_coefficients=field.pattern
            )
        
    @staticmethod
    def get_rabi_frequency_amplitude(driving: braket_ir.DrivingField):
        field = braket_to_quera.get_global_field(driving.amplitude)
        return RabiFrequencyAmplitude(
            **{"global":braket_to_quera.get_global_field(driving.amplitude)}
        )
        
    @staticmethod
    def get_rabi_frequency_phase(driving: braket_ir.DrivingField):
        return RabiFrequencyPhase(
            **{"global":braket_to_quera.get_global_field(driving.phase)}
        )
    
    @staticmethod
    def get_detuning(driving: braket_ir.DrivingField, shifting: Optional[braket_ir.ShiftingField]):
        if shifting == None:
            return Detuning(
                **{"global":braket_to_quera.get_global_field(driving.detuning)}
            )
        else:
            return Detuning(
                **{"global":braket_to_quera.get_global_field(driving.detuning),
                "local":braket_to_quera.get_local_field(shifting.magnitude)}
            )

    @staticmethod
    def get_rydberg(driving: braket_ir.DrivingField, shifting: Optional[braket_ir.ShiftingField] = None):
        return RydbergHamiltonian(
            rabi_frequency_amplitude = braket_to_quera.get_rabi_frequency_amplitude(driving),
            rabi_frequency_phase=braket_to_quera.get_rabi_frequency_phase(driving),
            detuning=braket_to_quera.get_detuning(driving, shifting)
        )


    @staticmethod
    def get_effective_hamiltonian(hamiltonian_ir: braket_ir.Hamiltonian):
        
        driving_fields = hamiltonian_ir.drivingFields
        shifting_fields = hamiltonian_ir.shiftingFields
        
        if len(driving_fields) != 1: raise ValueError("QuEra IR only supports exactly one set of driving fields")
        if len(shifting_fields) > 1:  raise ValueError("QuEra IR only supports at most one set of shifting fields")
        
        if len(shifting_fields) == 0:
            return EffectiveHamiltonian(
                rydberg=braket_to_quera.get_rydberg(driving_fields[0])
            )
        else:
            return EffectiveHamiltonian(
                rydberg=braket_to_quera.get_rydberg(driving_fields[0], shifting_fields[0])
            )

    @staticmethod
    def get_lattice(setup: braket_ir.Setup):
        return Lattice(sites=setup.ahs_register.sites, filling=setup.ahs_register.filling)

def quera_task_to_braket_ahs(task_specification: QuEraTaskSpecification) -> Tuple[int,AnalogHamiltonianSimulation]:
    """Convert a QuEra compatible program to a braket AHS program. 

    Args:
        js (QuEraTaskSpecification): An object containing a program formatted to be accepted by 
        the QuEra API. 

    Returns:
        Tuple[int,AnalogHamiltonianSimulation]: A tuple continaing
            the number of shots as the first element and the ahs program 
            as the second argument. 
    """

    return task_specification.nshots, AnalogHamiltonianSimulation(
            register=quera_to_braket.get_register(task_specification.lattice), 
            hamiltonian=quera_to_braket.get_hamiltonian(task_specification.effective_hamiltonian)
        )

def braket_ahs_to_quera_task(ahs : AnalogHamiltonianSimulation, shots: int = 1) -> QuEraTaskSpecification:
    """Translates Braket AHS IR program to Quera-TaskSpecification object.

    Args:
        ahs (AnalogHamiltonianSimulation): AHS object from braket SDK
        shots (int): The number of shots to run this program

    Returns:
        QuEraTaskSpecification:  QuEraTaskSpecification object representation of program
    """

    ahs_ir = ahs.to_ir()
    # QuEra IR Program
    return QuEraTaskSpecification(
        nshots=shots,
        lattice=braket_to_quera.get_lattice(ahs_ir.setup),
        effective_hamiltonian=braket_to_quera.get_effective_hamiltonian(ahs_ir.hamiltonian)
    )
