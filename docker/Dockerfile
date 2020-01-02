FROM paddyroddy/cosmoinformatics

WORKDIR /home/s2sleplet

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ls -a

WORKDIR pys2sleplet

RUN ls -a

WORKDIR data

RUN ls -a

WORKDIR maps

RUN ls -a

WORKDIR earth

RUN ls -a

# above 512 get a memory error from travis
RUN sed -i 's/16/512/' pys2sleplet/config.toml

RUN pytest -v