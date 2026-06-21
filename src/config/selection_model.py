from pydantic import BaseModel


class SelectionConfig(BaseModel):
    """Configuration for the policy selection pipeline.

    Args:
        procedural_variance: Noise injected into weighted choices to add diversity (0.0–1.0).
        timeout_limit_sec: Maximum wall-clock seconds for a single selection decision.
        async_batch_size: Number of concurrent Showdown battles in a single Oracle expansion wave.
    """

    procedural_variance: float = 0.20
    timeout_limit_sec: float = 60.0
    async_batch_size: int = 4
