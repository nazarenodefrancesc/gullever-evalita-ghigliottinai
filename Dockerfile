FROM ubuntu:18.04

MAINTAINER Nazareno De Francesco "nazareno.defrancesco@celi.it"

RUN apt update -y && \
    apt install -y python3-pip python3-dev

WORKDIR /evalita-ghigliottinai

ADD . /evalita-ghigliottinai

RUN python3 setup.py install

ENTRYPOINT [ "python3" ]

CMD [ "server_rest.py" ]
