import numpy as np
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import AnalogHamiltonianSimulationQuantumTaskResult
from typing import Dict

__all__ = [
    'get_counts',
    'get_avg_density',
]


def get_counts(result: AnalogHamiltonianSimulationQuantumTaskResult) -> Dict[str, int]:
    """Aggregate state counts from AHS shot results

        Args:
            result (AnalogHamiltonianSimulationQuantumTaskResult): The result 
                from which the aggregated state counts are obtained

        Returns:
            Dict[str, int]: number of times each state configuration is measured

        Notes: We use the following convention to denote the state of an atom (site):
            e: empty site
            r: Rydberg state atom
            g: ground state atom
    """

    state_counts = Counter()
    states = ['e', 'r', 'g']
    for shot in result.measurements:
        pre = shot.pre_sequence
        post = shot.post_sequence
        state_idx = np.array(pre) * (1 + np.array(post))
        state = "".join(map(lambda s_idx: states[s_idx], state_idx))
        state_counts.update((state,))

    return dict(state_counts)

def get_avg_density(result: AnalogHamiltonianSimulationQuantumTaskResult) -> np.ndarray:
    """Get the average Rydberg densities from the result

        Args:
            result (AnalogHamiltonianSimulationQuantumTaskResult): The result 
                from which the aggregated state counts are obtained

        Returns: 
            ndarray: The average densities from the result
    """    

    measurements = result.measurements
    postSeqs = [measurement.post_sequence for measurement in measurements]
    postSeqs = 1 - np.array(postSeqs) # change the notation such 1 for rydberg state, and 0 for ground state
    
    avg_density = np.sum(postSeqs, axis=0)/len(postSeqs)
    
    return avg_density


