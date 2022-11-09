
from itertools import product
import json
import numpy as np

def parallelize_quera_json(input_json,interproblem_distance,qpu_width,qpu_height,n_site_max):
    input_json
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




def get_shots_quera_results(results_json: dict,batch_mapping=None,post_select=True):
    # collecting QPU Data
    no_defects = lambda bits: np.all(bits==1) if post_select else True
    shots_list = results_json["shot_outputs"]
    
    pre_and_post = [(np.array(m["pre_sequence"]),np.array(m["post_sequence"])) for m in shots_list  if m["shot_status_code"]==200]


    if batch_mapping == None:
        all_sequences = [post for pre,post in pre_and_post  if no_defects(pre)]
    else:
        all_sequences = [post[inds] for (pre,post),inds in product(pre_and_post,batch_mapping.values()) if no_defects(pre[inds])]

    return np.array(all_sequences)


