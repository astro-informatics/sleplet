ARG DOCKER_USERNAME
ARG DOCKER_IMAGE

FROM ${DOCKER_USERNAME}/${DOCKER_IMAGE}

ENV S2SLEPLET /home/project/src/s2sleplet

WORKDIR ${S2SLEPLET}

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
