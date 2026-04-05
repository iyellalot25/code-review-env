from server.tasks.base_task import BaseTask


class HardTask(BaseTask):
    def __init__(self):
        super().__init__("fixtures/hard_pr.json")

    @property
    def task_name(self) -> str:
        return "hard-review"

    @property
    def max_steps(self) -> int:
        return 12
