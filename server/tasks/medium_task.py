from server.tasks.base_task import BaseTask


class MediumTask(BaseTask):
    def __init__(self):
        super().__init__("fixtures/medium_pr.json")

    @property
    def task_name(self) -> str:
        return "medium-review"

    @property
    def max_steps(self) -> int:
        return 10
