FROM pytorch/pytorch

WORKDIR /tvtuner
RUN pip install "tvtuner[dev] @ git+https://github.com/dewcservices/tvtuner.git"

WORKDIR /app
COPY *.pkl .
COPY service.py .

EXPOSE 8000
ENTRYPOINT uvicorn service:app
