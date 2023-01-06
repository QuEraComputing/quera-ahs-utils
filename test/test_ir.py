import json
import unittest
import sys
import os

from braket.ahs.atom_arrangement import SiteType
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement,SiteType


from quera_ahs_utils.ir import braket_sdk_to_quera_json, quera_json_to_ahs
import numpy as np



class IrModule(unittest.TestCase):

    def pulse_values(self):
        # Define a set of time points
        time_points = [0, 0.6e-6, 3.4e-6, 4e-6]

        # Define the strength of the transverse field Ω
        amplitude_min = 0
        amplitude_max = 10e6  # rad / sec

        # Define the strength of the detuning Δ
        Delta_initial = -20e6     # rad / sec
        Delta_final = 20e6 

        amplitude_values = [amplitude_min, amplitude_max, amplitude_max, amplitude_min]  # piecewise linear
        detuning_values = [Delta_initial, Delta_initial, Delta_final, Delta_final]  # piecewise linear
        phase_values = [0, 0, 0, 0]  # piecewise constant

        return time_points,amplitude_values,detuning_values,phase_values


    def generate_ahs_program(self):
        nshots = 100
        register = AtomArrangement()
        register.add((0.0,0.0),SiteType.FILLED)
        register.add((0.0,5.0e-6),SiteType.FILLED)
        register.add((0.0,1e-5),SiteType.VACANT)

        time_points,amplitude_values,detuning_values,phase_values = self.pulse_values()

        amplitude = TimeSeries()
        for t,v in zip(time_points,amplitude_values):
            amplitude.put(t,v)

        detuning = TimeSeries()
        for t,v in zip(time_points,detuning_values):
            detuning.put(t,v)

        phase = TimeSeries()
        for t,v in zip(time_points,phase_values):
            phase.put(t,v)

        drive = DrivingField(
            amplitude=amplitude, 
            detuning=detuning, 
            phase=phase
        )

        return nshots,AnalogHamiltonianSimulation(
            register=register, 
            hamiltonian=drive
        )

    def generate_quera_ir(self):
        nshots = 100

        sites = [
            [0.0,0.0],
            [0.0,5.0e-6],
            [0.0,1.0e-5],
        ]
        filling = [1,1,0]

        time_points,amplitude_values,detuning_values,phase_values = self.pulse_values()

        # QuEra IR Program
        quera_ir = dict()
        quera_ir["nshots"] = nshots
        quera_ir["lattice"] = {"sites":sites,"filling":filling}
        quera_ir["effective_hamiltonian"] = {
            "rydberg": {
                "rabi_frequency_amplitude": {"global": {"times": time_points, "values": amplitude_values}},
                "rabi_frequency_phase": {"global": {"times": time_points, "values":phase_values}},
                "detuning": {
                    "global": {"times": time_points, "values": detuning_values},
                },
            }
        }

        return quera_ir

    def test_quera_to_braket(self):
        nshots,ahs_program = self.generate_ahs_program()
        quera_ir = self.generate_quera_ir()
        translated_quera_ir = braket_sdk_to_quera_json(ahs_program,shots=nshots)
        self.assertDictEqual(translated_quera_ir,quera_ir)

    def test_braket_to_quera(self):
        nshots,ahs_program = self.generate_ahs_program()
        quera_ir = self.generate_quera_ir()
        translated_nshots,translated_ahs_program = quera_json_to_ahs(quera_ir)
        self.assertEqual(ahs_program.to_ir().json(),translated_ahs_program.to_ir().json())
        self.assertEqual(nshots,translated_nshots)



tester = IrModule()

tester.test_braket_to_quera()
