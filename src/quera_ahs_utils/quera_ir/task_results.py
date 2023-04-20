from enum import Enum
from pydantic import BaseModel, conlist, conint
from typing import Callable, Optional, List, Tuple
import numpy as np

__all__ = [
    "QuEraTaskResults",  
    "TaskProbabilities"  
]

class QuEraShotStatusCode(str, Enum):
    Completed = "Completed"
    MissingPreSequence = "MissingPreSequence"
    MissingPostSequence = "MissingPostSequence"
    MissingMeasurement = "MissingMeasurement"

class QuEraTaskStatusCode(str, Enum):
    Created = "Created"
    Running = "Running"
    Completed = "Completed"
    Failed = "Failed"
    Cancelled = "Cancelled"

class QuEraShotResult(BaseModel):
    shot_status: QuEraShotStatusCode = QuEraShotStatusCode.MissingMeasurement
    pre_sequence: conlist(conint(ge=0, le=1), min_items=0) = []
    post_sequence: conlist(conint(ge=0, le=1), min_items=0) = []

class TaskProbabilities(BaseModel):
    probabilities: List[Tuple[Tuple[str,str],float]]
    
    def simulate_task_result(self, shots = 1):
        bit_strings, probabilities = zip(*self.probabilities)
        
        indices = np.random.choice(
            len(probabilities), 
            p=probabilities,
            size=shots
        )
        shot_outputs = []
        for index in indices:
            pre_string, post_string = bit_strings[index]
            pre_sequence = [int(bit) for bit in pre_string]
            post_sequence = [int(bit) for bit in post_string]
            
            shot_outputs.append(
                QuEraShotResult(
                    shot_status = QuEraShotStatusCode.Completed,
                    pre_sequence = pre_sequence,
                    post_sequence = post_sequence
                )
            )
        
        return QuEraTaskResults(
            task_status=QuEraTaskStatusCode.Completed,
            shot_outputs=shot_outputs
        )

class QuEraTaskResults(BaseModel):
    task_status: QuEraTaskStatusCode = QuEraTaskStatusCode.Failed
    shot_outputs: conlist(QuEraShotResult, min_items=0) = []
    
    def export_as_probabilties(self, post_process=False) -> TaskProbabilities:
        """converts from shot results to probabilities

        Returns:
            TaskProbabilities: The task results as probabilties
        """
        probabilities = dict()
        n = 0
        for shot_result in self.shot_outputs:
            if any(bit==0 for bit in shot_result.pre_sequence):
                continue
            
            pre_sequence_str = "".join(
                str(bit) for bit in shot_result.pre_sequence
            )
            
            post_sequence_str = "".join(
                str(bit) for bit in shot_result.post_sequence
            )
            
            configuration = (pre_sequence_str,post_sequence_str)
            # iterative average
            probabilities[configuration] = \
                ((n + 1.0) * probabilities.get(configuration, 0) + 1.0)/n
                
            n += 1
            
        return TaskProbabilities(list(probabilities.items()))
    
    def post_process(self, keep_shot_result: Optional[Callable] = None):
        
        if keep_shot_result == None:
            keep_shot_result = \
                lambda shot_result: \
                    any(bit==0 for bit in shot_result.pre_sequence)
        
        shot_outputs = []
        for shot_result in self.shot_outputs:
            if not keep_shot_result(shot_result):
                continue
            
            shot_outputs.append(shot_result)
            
        return QuEraTaskResults(
            task_status=self.task_status,
            shot_outputs=shot_outputs
        )
