FROM python:3.11.3-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry

ENV POETRY_CACHE_DIR=/poetry_cache

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN poetry config virtualenvs.create false \
    && for i in 1 2 3; do poetry install --no-root && break || sleep 10; done

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8000"]