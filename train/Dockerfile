FROM tjems6498/surface_pipeline_base:2 AS builder

RUN pip install -U numpy albumentations tqdm timm mlflow boto3

ADD . /app/

WORKDIR /app
ENTRYPOINT ["python", "train.py" ]
