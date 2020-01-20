ARG DOCKER_USERNAME
ARG DOCKER_IMAGE

FROM ${DOCKER_USERNAME}/${DOCKER_IMAGE}

WORKDIR /home/project/src/s2sleplet

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /home/project/src/s2sleplet/pys2sleplet

# above 512 get a memory error from travis
RUN sed -i 's/16/512/' config/settings.toml

RUN pytest -v
