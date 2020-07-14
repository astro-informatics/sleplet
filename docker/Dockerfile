ARG DOCKER_USERNAME
ARG DOCKER_IMAGE

# hadolint ignore=DL3006
FROM ${DOCKER_USERNAME}/${DOCKER_IMAGE}

WORKDIR /home/project/src/s2sleplet

# required for gmpy2
RUN apt-get install -y --no-install-recommends \
    libgmp-dev \
    libmpc-dev \
    libmpfr-dev

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
