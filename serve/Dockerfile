FROM tjems6498/surface_pipeline_base:3 AS builder

RUN pip install -U pandas scikit-learn mlflow boto3 bentoml protobuf~=3.19.0
RUN mkdir -p /app

ADD . /app/

WORKDIR /app
ENTRYPOINT ["python", "bento_packer.py" ]