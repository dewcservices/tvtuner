FROM pytorch/pytorch

WORKDIR /tvtuner
RUN apt-get update && apt-get install git -y
RUN pip install "tvtuner[dev] @ git+https://github.com/dewcservices/tvtuner.git"

WORKDIR /app
COPY *.pkl .
COPY service.py .

EXPOSE 8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
