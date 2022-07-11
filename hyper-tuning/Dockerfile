FROM tjems6498/surface_pipeline_base AS builder

RUN pip install -U numpy albumentations wandb tqdm timm

ADD . /app/
WORKDIR /app
ENTRYPOINT ["python3", "sweep.py"]

