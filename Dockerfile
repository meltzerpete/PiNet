FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN apt-get update && \
apt-get install -y unzip ranger vim-gtk3

ENV GAMMA_DATA_ROOT /app/Data

RUN conda install scikit-learn keras numpy tensorflow tensorboard pandas networkx matplotlib

CMD wget 'https://www.dropbox.com/sh/fdjopjwwpoff149/AABWjYu08HDNXWAbBeACSEzMa?dl=0' -O ImportData.zip; \
mkdir ImportData; \
unzip ImportData.zip -d ImportData; \
rm -rf ImportData.zip; \
wget 'https://www.dropbox.com/sh/6bi7ynmcxs0s7ha/AAAuxD1QVs3DtmOoZuskvTLxa?dl=0' -O Data.zip; \
mkdir Data; \
unzip Data.zip -d Data; \
rm -rf Data.zip; \
python model/ClassificationAccuracyTimeBenchmark.py
