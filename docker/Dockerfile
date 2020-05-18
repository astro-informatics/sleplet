ARG DOCKER_USERNAME
ARG DOCKER_IMAGE

FROM ${DOCKER_USERNAME}/${DOCKER_IMAGE}:latest

WORKDIR /home/project/src/s2sleplet

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
