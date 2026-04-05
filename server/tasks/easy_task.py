from server.tasks.base_task import BaseTask


class EasyTask(BaseTask):
    def __init__(self):
        super().__init__("fixtures/easy_pr.json")

    @property
    def task_name(self) -> str:
        return "easy-review"

    @property
    def max_steps(self) -> int:
        return 8
