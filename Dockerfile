FROM pytorch/pytorch:latest

RUN mkdir -p /app
COPY ./utils/ /app/
