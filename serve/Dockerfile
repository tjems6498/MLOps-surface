FROM bitnami/pytorch:1.11.0

RUN pip install -U mlflow boto3 protobuf~=3.19.0 bentoml
RUN mkdir -p /app

COPY . /app/

WORKDIR /app
ENTRYPOINT ["python", "bento_packer.py" ]