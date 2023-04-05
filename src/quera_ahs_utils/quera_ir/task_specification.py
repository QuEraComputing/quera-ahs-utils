from pydantic import BaseModel, conint, conlist
from typing import Optional, List, Tuple, Union
from decimal import Decimal


__all__ = [
    "QuEraTaskSpecification"
]

FloatType = Union[Decimal, float]

class GlobalField(BaseModel):
    times: conlist(FloatType, min_itens=2, unique_items=True)
    values: conlist(FloatType, min_items=2, unique_items=True)
    
    def __hash__(self):
        return hash((GlobalField, tuple(self.times), tuple(self.values)))
    
class LocalField(BaseModel):
    times: conlist(FloatType, min_itens=2, unique_items=True)
    values: conlist(FloatType, min_items=2, unique_items=True)
    lattice_site_coefficients: conlist(FloatType, min_items=1)
    
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
    
class RydbergHamiltonian(BaseModel):
    rabi_frequency_amplitude: RabiFrequencyAmplitude
    rabi_frequency_phase: RabiFrequencyPhase
    detuning: Detuning
    
    def __hash__(self):
        return hash((RydbergHamiltonian, self.rabi_frequency_amplitude, self.rabi_frequency_phase, self.detuning))
    
class EffectiveHamiltonian(BaseModel):
    rydberg: RydbergHamiltonian
    
    def __hash__(self):
        return hash((EffectiveHamiltonian, self.rydberg))
    
class Lattice(BaseModel):
    sites: conlist(Tuple[FloatType, FloatType], min_items=1)
    filling: conlist(int, min_items=1)
    
    def __hash__(self):
        return hash((Lattice, tuple(self.sites), tuple(self.filling)))
    
class QuEraTaskSpecification(BaseModel):
    nshots: conint(ge=1, le=1000)
    lattice: Lattice
    effective_hamiltonian: EffectiveHamiltonian
    
    def __hash__(self):
        return hash((QuEraTaskSpecification, self.nshots, self.lattice, self.effective_hamiltonian))
    
