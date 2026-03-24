FROM --platform=linux/amd64 python:3.7-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ghigliottinai

COPY ./requirements-docker.txt /srv/ghigliottinai/
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY ./server_rest_lite.py /srv/ghigliottinai
COPY ./config.py /srv/ghigliottinai
COPY ./setup.py /srv/ghigliottinai
COPY ./logging_conf.py /srv/ghigliottinai
COPY ./resources /srv/ghigliottinai/resources
COPY ./texmega /srv/ghigliottinai/texmega
COPY ./cooccurrence_matrix /srv/ghigliottinai/cooccurrence_matrix

# Decompress all .pkl.zip archives and remove the zips
RUN find /srv/ghigliottinai -name "*.pkl.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;

EXPOSE 9000

ENTRYPOINT ["uwsgi", \
        "--master", \
        "--lazy", \
        "--http-socket", "0.0.0.0:9000", \
        "--callable", "app", \
        "--wsgi-file", "/srv/ghigliottinai/server_rest_lite.py"]

CMD []