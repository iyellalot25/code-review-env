import os
import uvicorn
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

try:
    from server.env import CodeReviewEnv
    from server.models import CodeReviewAction, ResetResult, StepResult, StateResult
except ImportError:
    from env import CodeReviewEnv
    from models import CodeReviewAction, ResetResult, StepResult, StateResult

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


class ResetRequest(BaseModel):
    task: Optional[str] = None


@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    global _env
    # Allow task to be passed in the request body OR via env var
    if request and request.task:
        os.environ["CODE_REVIEW_TASK"] = request.task
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


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Code Review Assistant",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


def main():
    """Entry point for the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()