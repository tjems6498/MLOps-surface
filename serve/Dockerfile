FROM pytorch/pytorch:latest

RUN apt-get -y update && apt-get install -y libzbar-dev
RUN pip install -U mlflow boto3 protobuf~=3.19.0 bentoml timm
RUN mkdir -p /app

COPY . /app/

WORKDIR /app
ENTRYPOINT ["python", "bento_packer.py" ]