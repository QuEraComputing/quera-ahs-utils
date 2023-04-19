from enum import Enum
from pydantic import BaseModel, conlist, conint
from typing import Dict, Tuple

__all__ = [
    "QuEraTaskResults",    
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
    probabilities: Dict[Tuple[str,str], float]

class QuEraTaskResults(BaseModel):
    task_status: QuEraTaskStatusCode = QuEraTaskStatusCode.Failed
    shot_outputs: conlist(QuEraShotResult, min_items=0) = []
    
    def export_as_probabilties(self) -> TaskProbabilities:
        """converts from shot results to probabilities

        Returns:
            TaskProbabilities: The task results as probabilties
        """
        probabilities = dict()
        n_shots = len(self.shot_outputs)
        for shot_result in self.shot_outputs:
            
            pre_sequence_str = "".join(
                str(bit) for bit in shot_result.pre_sequence
            )
            
            post_sequence_str = "".join(
                str(bit) for bit in shot_result.post_sequence
            )
            
            configuration = (pre_sequence_str,post_sequence_str)
            probabilities[configuration] = \
                probabilities.get(configuration, 0) + 1.0/n_shots
                
        return TaskProbabilities(probabilities)
