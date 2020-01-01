FROM paddyroddy/cosmoinformatics
LABEL maintainer="paddyroddy.github.io"

WORKDIR /home/s2sleplet

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# above 512 get a memory error from travis
RUN sed -i 's/16/512/' pys2sleplet/config.toml

RUN pytest -v