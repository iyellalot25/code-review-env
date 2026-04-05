from abc import ABC, abstractmethod
from server.models import IssueReport
from typing import List
import json


class BaseTask(ABC):
    def __init__(self, fixture_path: str):
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        self.ground_truth: List[IssueReport] = [
            IssueReport(**i) for i in self.fixture["ground_truth_issues"]
        ]

    @property
    def pr_title(self) -> str:
        return self.fixture["pr_title"]

    @property
    def pr_description(self) -> str:
        return self.fixture["pr_description"]

    @property
    def diff(self) -> str:
        return self.fixture["diff"]

    @property
    def file_contents(self) -> dict:
        return self.fixture["files"]

    @property
    @abstractmethod
    def task_name(self) -> str: ...

    @property
    @abstractmethod
    def max_steps(self) -> int: ...
