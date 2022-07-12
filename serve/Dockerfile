FROM pytorch/pytorch:latest

RUN pip install -U mlflow boto3 bentoml protobuf~=3.19.0
RUN mkdir -p /app

COPY . /app/

WORKDIR /app
ENTRYPOINT ["python", "bento_packer.py" ]