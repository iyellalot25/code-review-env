from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any


class IssueReport(BaseModel):
    issue_type: Literal["bug", "security", "performance", "style", "logic"]
    line_number: Optional[int] = None
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    suggested_fix: Optional[str] = None


class CodeReviewAction(BaseModel):
    action_type: Literal["flag_issue", "approve", "request_changes", "add_comment"]
    issues: Optional[List[IssueReport]] = None
    comment: Optional[str] = None
    final_verdict: Optional[Literal["approve", "request_changes", "reject"]] = None


class CodeReviewObservation(BaseModel):
    pr_title: str
    pr_description: str
    diff: str
    file_contents: Dict[str, str]
    previous_actions: List[str]
    step_number: int
    feedback: Optional[str] = None
    done: bool = False


class StepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    observation: CodeReviewObservation


class StateResult(BaseModel):
    task_name: str
    step_number: int
    total_reward: float
    done: bool
    issues_found: List[IssueReport]
    ground_truth_count: int
