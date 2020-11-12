FROM ubuntu:18.04

RUN apt-get update && apt-get install -y uwsgi uwsgi-plugin-python3 python3 python3-pip

WORKDIR /srv/ghigliottinai
COPY ./server_rest_lite.py /srv/ghigliottinai
COPY ./config.py /srv/ghigliottinai
COPY ./setup.py /srv/ghigliottinai
COPY ./logging_conf.py /srv/ghigliottinai
COPY ./resources /srv/ghigliottinai/resources
COPY ./texmega /srv/ghigliottinai/texmega
COPY ./cooccurrence_matrix /srv/ghigliottinai/cooccurrence_matrix

RUN useradd uwsgi
RUN pip3 install --user -e /srv/ghigliottinai

ENTRYPOINT ["uwsgi", \
        "--master", \
        "--lazy", \
        "--http-socket", "0.0.0.0:9000", \
        "--plugins", "python3", \
        "--callable", "app", \
        "--wsgi-file", "/srv/ghigliottinai/server_rest_lite.py"]

CMD []