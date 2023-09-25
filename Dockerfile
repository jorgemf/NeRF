FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN pip install poetry
COPY poetry.lock /tmp/
COPY pyproject.toml /tmp/
WORKDIR /tmp
RUN poetry export -f requirements.txt  --without-hashes --output /tmp/requirements.txt
RUN sed -i '/^torch==/d' /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /project
ENV HOME=/project
