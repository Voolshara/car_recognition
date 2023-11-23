# FROM tensorflow/tensorflow:latest-gpu
# FROM tensorflow/tensorflow
FROM python:3.11

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN apt-get -y update \
 && apt-get -y upgrade 

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install poetry
RUN poetry config virtualenvs.create false
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx


COPY . .
RUN poetry install