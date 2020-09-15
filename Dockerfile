FROM python:3.6.1-alpine

MAINTANER Your Name "nazareno.defrancesco@celi.it"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

WORKDIR /evalita-ghigliottinai

ADD . /evalita-ghigliottinai

RUN python setup.py install

ENTRYPOINT [ "python" ]

CMD [ "server_rest.py" ]