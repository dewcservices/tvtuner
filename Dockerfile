FROM pytorch/pytorch

WORKDIR /tvtuner
COPY tvtuner/ .
COPY pyproject.toml .
COPY README.md .
RUN pip install .[api]

WORKDIR /app
COPY examples/*.pkl .
COPY examples/service.py .

EXPOSE 8000
ENTRYPOINT uvicorn service:app
