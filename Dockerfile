FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir numpy pandas pyarrow matplotlib tqdm

COPY model/ ./model/
COPY libs/cache.py libs/metrics.py libs/viz.py ./libs/
COPY train.py ./

ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
