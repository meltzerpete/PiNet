FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda install scikit-learn keras numpy tensorflow tensorboard pandas networkx matplotlib

RUN apt-get update && \
apt-get install -y unzip ranger vim-gtk3

RUN wget 'https://www.dropbox.com/sh/fdjopjwwpoff149/AABWjYu08HDNXWAbBeACSEzMa?dl=0' -O ImportData.zip && \
mkdir ImportData && \
unzip ImportData.zip -d ImportData && \
rm ImportData.zip
