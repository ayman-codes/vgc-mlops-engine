from pydantic import BaseModel


class SelectionConfig(BaseModel):
    procedural_variance: float = 0.20
    timeout_limit_sec: float = 60.0
    async_batch_size: int = 4
