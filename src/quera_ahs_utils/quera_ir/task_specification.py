from pydantic import BaseModel
from typing import Optional, List, Tuple, Union
from decimal import Decimal

from quera_ahs_utils.quera_ir.capabilities import QuEraCapabilities


__all__ = [
    "QuEraTaskSpecification"
]

# TODO: add version to these models.

FloatType = Union[Decimal, float]

def discretize_list(list_of_values: list, resolution: FloatType):
    resolution = Decimal(str(float(resolution)))
    return [Decimal(value).quantize(resolution) for value in list_of_values]

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
        allow_population_by_field_name = True
        fields = {
            'global_': 'global'
        }
    
    def __hash__(self):
        return hash((RabiFrequencyAmplitude, self.global_))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        global_time_resolution = task_capabilities.capabilities.rydberg.global_.time_resolution
        global_value_resolution =  task_capabilities.capabilities.rydberg.global_.rabi_frequency_resolution
        
        return RabiFrequencyAmplitude(
            global_ = GlobalField(
                times = discretize_list(self.global_.times, global_time_resolution),
                values = discretize_list(self.global_.values, global_value_resolution)
            ) 
        )

class RabiFrequencyPhase(BaseModel):
    global_: GlobalField
    
    class Config:
        allow_population_by_field_name = True
        fields = {
            'global_': 'global'
        }
        
    def __hash__(self):
        return hash((RabiFrequencyPhase, self.global_))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        global_time_resolution = task_capabilities.capabilities.rydberg.global_.time_resolution
        global_value_resolution =  task_capabilities.capabilities.rydberg.global_.phase_resolution
        
        return RabiFrequencyPhase(global_=GlobalField(
                    times = discretize_list(self.global_.times, global_time_resolution),
                    values = discretize_list(self.global_.values, global_value_resolution)
                )
            )
    
class Detuning(BaseModel):
    global_: GlobalField
    local: Optional[LocalField]
    
    class Config:
        allow_population_by_field_name = True
        fields = {
            'global_': 'global'
        }

    def __hash__(self):
        return hash((Detuning, self.global_, self.local))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        global_time_resolution = task_capabilities.capabilities.rydberg.global_.time_resolution
        global_value_resolution =  task_capabilities.capabilities.rydberg.global_.detuning_resolution
        local_time_resolution = task_capabilities.capabilities.rydberg.local.time_resolution

        if self.local != None:
            self.local = LocalField(
                    times = discretize_list(self.local.times, local_time_resolution), 
                    values = self.local.values,
                    lattice_site_coefficients=self.local.lattice_site_coefficients
                )


        return Detuning(
                global_ = GlobalField(
                    times = discretize_list(self.global_.times, global_time_resolution),
                    values = discretize_list(self.global_.values, global_value_resolution)
                ),
                local = self.local
            )
    
class RydbergHamiltonian(BaseModel):
    rabi_frequency_amplitude: RabiFrequencyAmplitude
    rabi_frequency_phase: RabiFrequencyPhase
    detuning: Detuning
    
    def __hash__(self):
        return hash((RydbergHamiltonian, self.rabi_frequency_amplitude, self.rabi_frequency_phase, self.detuning))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        return RydbergHamiltonian(
            rabi_frequency_amplitude = self.rabi_frequency_amplitude.discretize(task_capabilities),
            rabi_frequency_phase = self.rabi_frequency_phase.discretize(task_capabilities),
            detuning = self.detuning.discretize(task_capabilities)
        )

    
class EffectiveHamiltonian(BaseModel):
    rydberg: RydbergHamiltonian
    
    def __hash__(self):
        return hash((EffectiveHamiltonian, self.rydberg))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        return EffectiveHamiltonian(rydberg = self.rydberg.discretize(task_capabilities))

class Lattice(BaseModel):
    sites: List[Tuple[FloatType, FloatType]]
    filling: List[int]
    
    def __hash__(self):
        return hash((Lattice, tuple(self.sites), tuple(self.filling)))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        position_resolution = task_capabilities.capabilities.lattice.geometry.position_resolution
        return Lattice(
            sites = [discretize_list(site,position_resolution) for site in self.sites],
            filling = self.filling
        )
            
    
class QuEraTaskSpecification(BaseModel):
    nshots: int
    lattice: Lattice
    effective_hamiltonian: EffectiveHamiltonian
    
    def __hash__(self):
        return hash((QuEraTaskSpecification, self.nshots, self.lattice, self.effective_hamiltonian))
    
    def discretize(self, task_capabilities: QuEraCapabilities):
        return QuEraTaskSpecification(
            nshots = self.nshots,
            lattice = self.lattice.discretize(task_capabilities),
            effective_hamiltonian = self.effective_hamiltonian.discretize(task_capabilities)
        )
    
