
import dataclasses
from itertools import product
import numpy as np
from typing import Tuple,Union, NoReturn, Optional
from decimal import Decimal

from braket.aws import AwsDevice
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.field import Field
from braket.task_result import AnalogHamiltonianSimulationTaskResult
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.pattern import Pattern



__all__ = [
    'generate_parallel_register',
    'parallelize_field',
    'parallelize_hamiltonian',
    'parallelize_ahs',
    'get_shots_braket_sdk_results',
    'parallelize_quera_json',
    'get_shots_quera_results',
]

def generate_parallel_register(
    register:AtomArrangement,
    qpu:AwsDevice,
    interproblem_distance:Union[float,Decimal]
) -> Tuple[AtomArrangement,dict]:
    """generate grid of parallel registers from a single register. 

    Args:
        register (AtomArrangement): The register set to parallelize
        qpu (AwsDevice): The QPU that will be running the parallel job
        interproblem_distance (UnionType[float,Decimal]): _description_

    Returns:
        _type_: _description_
    """
    if len(register) > 1:
        x_min = min(*[site.coordinate[0] for site in register])
        x_max = max(*[site.coordinate[0] for site in register])
        y_min = min(*[site.coordinate[1] for site in register])
        y_max = max(*[site.coordinate[1] for site in register])
    else:
        x_min = x_max = 0
        y_min = y_max = 0

    single_problem_width = x_max - x_min
    single_problem_height = y_max - y_min

    # get values from device capabilities
    field_of_view_width = qpu.properties.paradigm.lattice.area.width
    field_of_view_height = qpu.properties.paradigm.lattice.area.height
    n_site_max = qpu.properties.paradigm.lattice.geometry.numberSitesMax
    
    batch_mapping = dict()
    parallel_register = AtomArrangement()

    atom_number = 0 #counting number of atoms added
    
    ix = 0
    while True:
        x_shift = ix * (single_problem_width + interproblem_distance)
        # reached the maximum number of batches possible given n_site_max
        if atom_number + len(register) > n_site_max: break
        # reached the maximum number of batches possible along x-direction
        if x_shift + single_problem_width > field_of_view_width: break 
        
        iy = 0
        while True:
            y_shift = iy * (single_problem_height + interproblem_distance)
            # reached the maximum number of batches possible given n_site_max
            if atom_number + len(register) > n_site_max: break 
            # reached the maximum number of batches possible along y-direction
            if y_shift + single_problem_height > field_of_view_height: 
                ix += 1
                break

            atoms = []
            for site in register:
                new_coordinate = (
                    x_shift + site.coordinate[0], 
                    y_shift + site.coordinate[1]
                )
                parallel_register.add(new_coordinate,site.site_type)

                atoms.append(atom_number)

                atom_number += 1
            
            batch_mapping[(ix,iy)] = atoms
            iy += 1

    return parallel_register,batch_mapping


def parallelize_field(
    field:Field,
    batch_mapping:dict
) -> Field:
    """Generate parallel field from a batch_mapping

    Args:
        field (Field): the field to parallelize
        batch_mapping (dict): the mapping that describes the parallelization

    Returns:
        Field: the new field that works for the parallel program. 
    """
    if field.pattern == None or field.pattern == "uniform":
        return field
    else:
        natoms = sum([len(atom_list) for atom_list in batch_mapping.values()])

        parallel_pattern_series = np.empty(natoms, dtype=object)
        for atom_subset in batch_mapping.values():
            parallel_pattern_series[atom_subset] = field.pattern.series
        
        return Field(field.time_series, Pattern(list(parallel_pattern_series)))
        

def parallelize_hamiltonian(
    hamiltonian: Hamiltonian,
    batch_mapping:dict
) -> DrivingField:
    """Generate the parallel driving fields from a batch_mapping. 

    Args:
        hamiltonian (Hamiltonian): The fields to parallelize
        batch_mapping (dict): the mapping that generates the parallelization

    Returns:
        Hamiltonian: the parallelized driving field. 
    """
    if isinstance(hamiltonian, ShiftingField):
        return ShiftingField(parallelize_field(hamiltonian.magnitude))
    elif isinstance(hamiltonian, DrivingField):
        return DrivingField(
                    amplitude=parallelize_field(hamiltonian.amplitude, batch_mapping),
                    phase=parallelize_field(hamiltonian.phase, batch_mapping),
                    detuning=parallelize_field(hamiltonian.detuning, batch_mapping)
                )
    elif isinstance(hamiltonian, Hamiltonian):
        return Hamiltonian(
                    map(
                        lambda h: parallelize_hamiltonian(h, batch_mapping), 
                        hamiltonian.terms
                    )
                )
    else:
        raise ValueError("expecting Hamiltonian or subtype of Hamiltonian")


def parallelize_ahs(
    ahs:AnalogHamiltonianSimulation,
    qpu: AwsDevice,
    interproblem_distance: Union[float,Decimal]
) -> AnalogHamiltonianSimulation:
    """Generate parallel ahs program. 

    Args:
        ahs (AnalogHamiltonianSimulation): The program to parallelize
        qpu (AwsDevice): The device to run the parallel jobs
        interproblem_distance (float, Decimal): The distance between the programs. 

    Returns:
        AnalogHamiltonianSimulation: The new parallel program ready to run. 
    """
    parallel_register,batch_mapping = generate_parallel_register(ahs.register,qpu,interproblem_distance)

    parallel_program = AnalogHamiltonianSimulation(
        register=parallel_register,
        hamiltonian=parallelize_hamiltonian(ahs.hamiltonian,batch_mapping)
    )
    return parallel_program,batch_mapping


def get_shots_braket_sdk_results(
    results: AnalogHamiltonianSimulationTaskResult,
    batch_mapping:Optional[dict]=None,
    post_select:Optional[bool]=True
)-> np.array:
    """get the shot results from a braket-sdk task results type. 

    Args:
        results (AnalogHamiltonianSimulationTaskResult): 
            The task results to process. 
        batch_mapping (Optional[dict], optional): 
            The parallel mapping generated from some parallelization. Defaults to None.
        post_select (bool, optional): 
        Post select if atom fails to be sorted in the shot results. Defaults to True.

    Returns:
        np.array: The shot results stored as 1,0 in the rows of the array.
        1 is the rydberg state and 0 is the ground state.  
    """
    # collecting QPU Data
    has_defects = lambda x:  np.any(x==0) if post_select else True
    
    if batch_mapping == None:
        all_sequences = [m.post_sequence for m in results.measurements if has_defects(m.pre_sequence)]
    else:
        all_sequences = [m.post_sequence[inds] for m,inds in product(results.measurements,batch_mapping.values()) if has_defects(m.pre_sequence[inds])]
        
    return np.array(all_sequences)


def parallelize_quera_json(
    input_json: dict,
    interproblem_distance:float,
    qpu_width:float,
    qpu_height:float,
    n_site_max:int
) -> Tuple[dict,dict]:
    """Generate a parallel QuEra json program from a single program. 

    Args:
        input_json (dict): The input program to parallelize
        interproblem_distance (float): The distance between parallel problems
        qpu_width (float): The field of view width for the program
        qpu_height (float): The field of view height for the program
        n_site_max (int): Maximum number of sites allowed for a program. 

    Raises:
        NotImplementedError: local detuning currently not supported. 

    Returns:
        Tuple[dict,dict]: first element is the parallelized program as a dict. 
        The second element of the tuple is the batch mapping for post processing. 
    """

    lattice = input_json["lattice"]

    if "local" in input_json["effective_hamiltonian"]["rydberg"]["detuning"]:
        raise NotImplementedError("local detuning not supported in this function")

    else:
        output_json = {}
        output_json.update(input_json)

        sites = lattice["sites"]
        filling = lattice["filling"]
        if len(sites) > 1:
            xmin = min(*[site[0] for site in sites])
            xmax = max(*[site[0] for site in sites])
            ymin = min(*[site[1] for site in sites])
            ymax = max(*[site[1] for site in sites])

            single_problem_width = xmax-xmin
            single_problem_height = ymax-ymin
        else:
            single_problem_width = 0
            single_problem_height = 0


        n_width  = int(qpu_width  // (single_problem_width + interproblem_distance))
        n_height = int(qpu_height //  (single_problem_height + interproblem_distance))

        parallel_sites = []
        parallel_filling = []

        atom_number = 0 #counting number of atoms added
        batch_mapping = {}

        for ix in range(n_width):
            x_shift = np.around(ix * (single_problem_width + interproblem_distance),14)

            for iy in range(n_height):    
                y_shift = np.around(iy * (single_problem_height + interproblem_distance),14)

                # reached the maximum number of batches possible given n_site_max
                if atom_number + len(sites) > n_site_max: break 

                atoms = []
                for site,fill in zip(sites,filling):
                    new_site = [x_shift + site[0], y_shift + site[1]]

                    parallel_sites.append(new_site)
                    parallel_filling.append(fill)

                    atoms.append(atom_number)

                    atom_number += 1

                batch_mapping[f"({ix},{iy})"] = atoms


        output_json["lattice"]["sites"] = parallel_sites
        output_json["lattice"]["filling"] = parallel_filling

    
    return output_json,batch_mapping

def get_shots_quera_results(
    results_json: dict,
    batch_mapping:Optional[dict]=None,
    post_select:Optional[bool]=True
) -> np.array:
    """Get the shots out of a QuEra programming

    Args:
        results_json (dict): _description_
        batch_mapping (dict, optional): 
            The parallel mapping generated from some parallelization. Defaults to None.
        post_select (bool, optional): 
        Post select if atom fails to be sorted in the shot results. Defaults to True.

    Returns:
        np.array: The shot results stored as 1,0 in the rows of the array.
        1 is the rydberg state and 0 is the ground state.
    """
    # collecting QPU Data
    no_defects = lambda bits: np.all(bits==1) if post_select else True
    shots_list = results_json["shot_outputs"]
    
    pre_and_post = [(np.array(m["pre_sequence"]),np.array(m["post_sequence"])) for m in shots_list  if m["shot_status_code"]==200]


    if batch_mapping == None:
        all_sequences = [post for pre,post in pre_and_post  if no_defects(pre)]
    else:
        all_sequences = [post[inds] for (pre,post),inds in product(pre_and_post,batch_mapping.values()) if no_defects(pre[inds])]

    return np.array(all_sequences)
