FROM python:3.11.3-slim

WORKDIR /app/clovaai-craft/clovaai-craft-master

COPY pyproject.toml poetry.lock ./

RUN pip install poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . .

EXPOSE 8000

CMD ["uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8000"]
