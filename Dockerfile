FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY apps ./apps
COPY docs ./docs
COPY eval ./eval
COPY interfaces ./interfaces
COPY modules ./modules
COPY sim ./sim
COPY tests ./tests

RUN uv sync --frozen --group dev

CMD ["uv", "run", "python", "apps/run_task.py", "place the bottle on the tray"]
