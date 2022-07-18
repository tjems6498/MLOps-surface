FROM tjems6498/surface_pipeline_base:1 AS builder

RUN pip install -U numpy albumentations tqdm timm mlflow boto3 albumentations

ADD . /app/

WORKDIR /app
ENTRYPOINT ["python", "train.py" ]
