# Builder Stage
FROM python:3.11-slim AS Builder
WORKDIR /app

# install uv dependency
RUN pip install uv

# Copy dependecy manifest
COPY pyproject.toml uv.lock ./

# Synchronize dependencies into ve
RUN uv sync --frozen --no-dev

# RUNTIME
FROM python:3.11-slim
WORKDIR /app

# Transfer compiled ve
COPY --from=builder /app/.venv /app/.venv

# Prepend the ve to sys path
ENV PATH = "/app/.venv/bin:$PATH"

# Tranfer core app logic and configs
COPY src/ ./src/

# Default execution
CMD ["python", "-m", "src.agent.battle_policy.main"]
