from enum import Enum
from pydantic import BaseModel, conlist, conint

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

class QuEraTaskResults(BaseModel):
    task_status: QuEraTaskStatusCode = QuEraTaskStatusCode.Failed
    shot_outputs: conlist(QuEraShotResult, min_items=0) = []
    

