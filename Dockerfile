FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y curl git bzip2 ffmpeg g++

# pyenv, python
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH
RUN eval "$(pyenv init -)" && \
    pyenv install anaconda3-4.4.0 && \
    pyenv global anaconda3-4.4.0 && \
    pyenv rehash

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY models /app/
COPY yukari_direct_server /app/
COPY run.py /app/
COPY run.sh /app/

CMD bash run.sh
