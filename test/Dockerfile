FROM tjems6498/surface_pipeline_base:2 AS builder

RUN pip install -U numpy albumentations tqdm mlflow boto3 timm

ADD . /app/

WORKDIR /app
ENTRYPOINT ["python", "test.py" ]

