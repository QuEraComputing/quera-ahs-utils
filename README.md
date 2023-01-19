# quera-ahs-utils
This python package is a collection of tools that can be used to program QuEra's **neutral atom Analog Hamiltonian Simulator** (**ahs**). These tools are primarily targeted towards the usage of [Amazon's Braket quantum computing service](https://aws.amazon.com/braket/). The Braket Python SDK can be found [here](https://github.com/aws/amazon-braket-sdk-python) along with some examples of how to use their service through a collection of examples from both [Braket](https://github.com/aws/amazon-braket-examples/tree/main/examples/analog_hamiltonian_simulation) and [QuEra](https://github.com/QuEraComputing/QuEra-braket-examples).

Some of the source code contained in this package originates from [this](https://github.com/aws/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/ahs_utils.py) module which was co-authered by the Braket science team.

We would be remiss not to advertise our own [Julia](https://julialang.org/) SDK for programming QuEra's **ahs**, [Bloqade](https://queracomputing.github.io/Bloqade.jl/dev/) as well as modeling neutral atom quantum computing. 

## Installation
The package can be installed via `pip`:

```
    pip install quera-ahs-utils
```

## Package contents

**quera-ahs-utils** is broken up into 5 modules each dealing with specific tools summarized in the table below:
|              module             |                                                       description                                                      |
|:-------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| **quera_ahs_utils**.analysis    | Perform analysis on shot results                                                         |
| **quera_ahs_utils**.drive       | Easily generate different types of driving hamiltonians                                   |
| **quera_ahs_utils**.ir          | Transform between QuEra and Braket program representations                                |
| **quera_ahs_utils**.parallelize | Transform small jobs into a parallel set of jobs to maximize the field of view of the QPU |
| **quera_ahs_utils**.plotting    | Visualize both **ahs** programs as well as its results.                              |

A module reference can be found [here](https://queracomputing.github.io/quera-ahs-utils/)
