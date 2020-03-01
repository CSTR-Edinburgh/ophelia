FROM ubuntu:18.04

WORKDIR /srv/app

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN apt update
RUN apt install python-minimal -y
RUN apt install python-pip -y

RUN apt-get install libsndfile1 -y

RUN apt-get install -y flite

# Copy Source later to enable dependency caching
COPY requirements_CPU.txt /srv/app/
COPY requirements_MCD.txt /srv/app/
# This messing around is due to pip processing requirements in alphabetical order...
RUN pip2 install -r requirements_CPU.txt
RUN pip2 install -r requirements_MCD.txt
RUN pip2 install -r requirements_CPU.txt

COPY . /srv/app

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

CMD python2.7 ./server/server.py -c work/elhadji_01/elhadji_01.cfg