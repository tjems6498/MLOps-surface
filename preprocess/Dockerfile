FROM tjems6498/surface_pipeline_base:2 AS builder

RUN pip install -U numpy split-folders albumentations

ADD . /app/

WORKDIR /app
ENTRYPOINT [ "python", "preprocess.py" ]

