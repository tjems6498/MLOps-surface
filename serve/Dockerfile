FROM pytorch/pytorch:latest

RUN apt-get -y update && apt-get install -y libzbar-dev
RUN apt install -y netcat
RUN pip install -U mlflow boto3 protobuf~=3.19.0 bentoml==1.0.0 timm sqlalchemy==1.3.24 albumentations
RUN mkdir -p /app

COPY . /app/

WORKDIR /app
ENTRYPOINT ["python", "bento_push.py" ]
