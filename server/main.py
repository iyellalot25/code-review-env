from fastapi import FastAPI, HTTPException
from server.env import CodeReviewEnv
from server.models import CodeReviewAction, ResetResult, StepResult, StateResult

app = FastAPI(
    title="Code Review Assistant OpenEnv",
    description="An OpenEnv-compliant environment for AI-powered pull request code review.",
    version="1.0.0",
)

_env: CodeReviewEnv = None


def get_env() -> CodeReviewEnv:
    global _env
    if _env is None:
        _env = CodeReviewEnv()
    return _env


@app.post("/reset", response_model=ResetResult)
def reset():
    """Reset the environment and return the initial observation."""
    global _env
    _env = CodeReviewEnv()
    return _env.reset()


@app.post("/step", response_model=StepResult)
def step(action: CodeReviewAction):
    """Take a step in the environment by submitting a code review action."""
    env = get_env()
    return env.step(action)


@app.get("/state", response_model=StateResult)
def state():
    """Return the current state of the environment."""
    env = get_env()
    return env.state()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
