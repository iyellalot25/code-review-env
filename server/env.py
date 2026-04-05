from server.models import (
    CodeReviewObservation,
    CodeReviewAction,
    IssueReport,
    StepResult,
    ResetResult,
    StateResult,
)
from server.grader import grade
from server.tasks.easy_task import EasyTask
from server.tasks.medium_task import MediumTask
from server.tasks.hard_task import HardTask
from typing import List
import os

TASK_MAP = {
    "easy-review": EasyTask,
    "medium-review": MediumTask,
    "hard-review": HardTask,
}


class CodeReviewEnv:
    def __init__(self):
        task_name = os.getenv("CODE_REVIEW_TASK", "easy-review")
        if task_name not in TASK_MAP:
            raise ValueError(f"Unknown task: {task_name}. Must be one of: {list(TASK_MAP.keys())}")
        self.task = TASK_MAP[task_name]()
        self.step_number = 0
        self.all_reported_issues: List[IssueReport] = []
        self.previous_actions: List[str] = []
        self.total_reward = 0.0
        self.done = False
        self.last_grade: dict = {}

    def reset(self) -> ResetResult:
        self.step_number = 0
        self.all_reported_issues = []
        self.previous_actions = []
        self.total_reward = 0.0
        self.done = False
        self.last_grade = {}
        return ResetResult(observation=self._make_observation())

    def step(self, action: CodeReviewAction) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._make_observation(feedback="Episode already done."),
                reward=0.0,
                done=True,
                info=self.last_grade,
            )

        self.step_number += 1
        reward = 0.0
        feedback = ""

        if action.issues:
            self.all_reported_issues.extend(action.issues)
            grade_result = grade(self.all_reported_issues, self.task.ground_truth)
            self.last_grade = grade_result
            new_score = grade_result["score"]
            prev_score = self.total_reward
            reward = max(0.0, new_score - prev_score)
            self.total_reward = new_score
            feedback = (
                f"Graded: matched={grade_result['matched']}, "
                f"fp={grade_result['false_positives']}, "
                f"f1={grade_result['f1']:.3f}"
            )

        action_str = f"{action.action_type}"
        if action.issues:
            action_str += f"({len(action.issues)} issues)"
        self.previous_actions.append(action_str)

        is_final = (
            action.action_type in ("approve", "request_changes")
            or self.step_number >= self.task.max_steps
        )

        if is_final:
            self.done = True
            final_grade = grade(self.all_reported_issues, self.task.ground_truth)
            self.last_grade = final_grade
            reward = final_grade["score"]
            self.total_reward = final_grade["score"]
            feedback = (
                f"Final score: {final_grade['score']:.3f}, "
                f"recall={final_grade['recall']:.3f}, "
                f"precision={final_grade['precision']:.3f}"
            )

        return StepResult(
            observation=self._make_observation(feedback=feedback),
            reward=round(reward, 4),
            done=self.done,
            info=self.last_grade,
        )

    def state(self) -> StateResult:
        return StateResult(
            task_name=self.task.task_name,
            step_number=self.step_number,
            total_reward=self.total_reward,
            done=self.done,
            issues_found=self.all_reported_issues,
            ground_truth_count=len(self.task.ground_truth),
        )

    def _make_observation(self, feedback: str = "") -> CodeReviewObservation:
        return CodeReviewObservation(
            pr_title=self.task.pr_title,
            pr_description=self.task.pr_description,
            diff=self.task.diff,
            file_contents=self.task.file_contents,
            previous_actions=self.previous_actions,
            step_number=self.step_number,
            feedback=feedback if feedback else None,
            done=self.done,
        )
