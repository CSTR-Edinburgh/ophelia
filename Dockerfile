FROM ubuntu:18.04

WORKDIR /srv/app

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy Source later to enable dependency caching
COPY requirements_CPU.txt /srv/app/
RUN pip install -r requirements_CPU.txt

COPY . /srv/app

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

CMD python2.7 ./server/server.py -c work/elhadji_01/elhadji_01.cfg