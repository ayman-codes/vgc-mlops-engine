# Builder Stage
FROM python:3.14-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential
RUN pip install uv

# Enforce deterministic environment allocation
RUN python -m venv /app/.venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

# Resolve project dependencies into the active virtual environment
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python /app/.venv/bin/python
COPY . .

# Compile and inject framework into the identical virtual environment
RUN git clone https://gitlab.com/DracoStriker/pokemon-vgc-engine.git /tmp/vgc
RUN uv pip install --python /app/.venv/bin/python /tmp/vgc

# Runtime Stage
FROM python:3.14-slim
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY src/ ./src/

CMD ["python", "-m", "src.agent.battle_policy.main"]