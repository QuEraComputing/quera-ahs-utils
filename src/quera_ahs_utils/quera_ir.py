from pydantic import BaseModel
from typing import Optional, List, Tuple, Union
from decimal import Decimal


__all__ = [
    "GlobalField",
    "LocalField",
    "RabiFrequencyAmplitude",
    "RabiFrequencyPhase",
    "Detuning",
    "Rydberg",
    "EffectiveHamiltonian",
    "Lattice",
    "QuEraTaskSpecification"
]

FloatType = Union[Decimal, float]

class GlobalField(BaseModel):
    times: List[FloatType]
    values: List[FloatType]
    
    def __hash__(self):
        return hash((GlobalField, tuple(self.times), tuple(self.values)))
    
class LocalField(BaseModel):
    times: List[FloatType]
    values: List[FloatType]
    lattice_site_coefficients: List[FloatType]
    
    def __hash__(self):
        return hash((LocalField, tuple(self.times), tuple(self.values), tuple(self.lattice_site_coefficients)))
    
class RabiFrequencyAmplitude(BaseModel):
    global_: GlobalField
    
    class Config:
        fields = {
            'global_': 'global'
        }
    
    def __hash__(self):
        return hash((RabiFrequencyAmplitude, self.global_))

class RabiFrequencyPhase(BaseModel):
    global_: GlobalField
    
    class Config:
        fields = {
            'global_': 'global'
        }
        
    def __hash__(self):
        return hash((RabiFrequencyPhase, self.global_))
    
class Detuning(BaseModel):
    global_: GlobalField
    local: Optional[LocalField]
    
    class Config:
        fields = {
            'global_': 'global'
        }

    def __hash__(self):
        return hash((Detuning, self.global_, self.local))
    
class Rydberg(BaseModel):
    rabi_frequency_amplitude: RabiFrequencyAmplitude
    rabi_frequency_phase: RabiFrequencyPhase
    detuning: Detuning
    
    def __hash__(self):
        return hash((Rydberg, self.rabi_frequency_amplitude, self.rabi_frequency_phase, self.detuning))
    
class EffectiveHamiltonian(BaseModel):
    rydberg: Rydberg
    
    def __hash__(self):
        return hash((EffectiveHamiltonian, self.rydberg))
    
class Lattice(BaseModel):
    sites: List[Tuple[float, float]]
    filling: List[int]
    
    def __hash__(self):
        return hash((Lattice, tuple(self.sites), tuple(self.filling)))
    
class QuEraTaskSpecification(BaseModel):
    nshots: int
    lattice: Lattice
    effective_hamiltonian: EffectiveHamiltonian
    
    def __hash__(self):
        return hash((QuEraTaskSpecification, self.nshots, self.lattice, self.effective_hamiltonian))
    