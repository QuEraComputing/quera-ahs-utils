import json
import unittest
import sys,os


from quera_ahs_utils.parallelize import parallelize_quera_json,get_shots_quera_results
import numpy as np


def load_json(filename):
    with open(filename) as file:
        json_value = json.load(file)

    return json_value



class ParallelizeModule(unittest.TestCase):

    def test_1(self) -> None:
        cwd = os.path.dirname(os.path.abspath(__file__))

        input_file = load_json(os.path.join(cwd,"aux_files","test_1.json"))
        output_file = load_json(os.path.join(cwd,"aux_files","parallel_test_1.json"))
        result_file = load_json(os.path.join(cwd,"aux_files","test_1_results_1.json"))

        test_json,batch_mapping = parallelize_quera_json(input_file,1e-5,2.5e-5,2.5e-5,10)

        self.assertDictEqual(test_json,output_file)

        shots = get_shots_quera_results(result_file,batch_mapping=batch_mapping,post_select=True)

        actual_shots = np.array([[1],[0],[1],[0],[1],[0],[0]])
        self.assertTrue(np.all(shots==actual_shots))

        shots = get_shots_quera_results(result_file,batch_mapping=batch_mapping,post_select=False)
        actual_shots = np.array([[1],[0],[1],[0],[1],[0],[0],[0]])

        self.assertTrue(np.all(shots==actual_shots))


        self.assertTrue(np.all(shots==actual_shots))

    def test_2(self) -> None:
        cwd = os.path.dirname(os.path.abspath(__file__))
        result_file = load_json(os.path.join(cwd,"aux_files","test_1_results_1.json"))

        shots = get_shots_quera_results(result_file,batch_mapping=None,post_select=True)
        actual_shots = np.array([[1,0,1,0]])
        self.assertTrue(np.all(shots==actual_shots))

    def test_3(self) -> None:
        cwd = os.path.dirname(os.path.abspath(__file__))
        result_file = load_json(os.path.join(cwd,"aux_files","test_1_results_1.json"))

        shots = get_shots_quera_results(result_file,batch_mapping=None,post_select=False)
        actual_shots = np.array([[1,0,1,0],[1,0,0,0]])
        self.assertTrue(np.all(shots==actual_shots))

