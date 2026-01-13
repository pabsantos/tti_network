FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY main.py CLAUDE.md ./

RUN mkdir -p data/raw/tti_shape data/raw/od_zones data/output data/test cache

CMD ["uv", "run", "python", "main.py"]
