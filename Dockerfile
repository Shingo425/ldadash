# ベースイメージ
FROM python:3.7-buster

WORKDIR LDAdash
COPY LDAdash.py requirements.txt /LDAdash
COPY import_py /LDAdash/import_py
#ADD  mecab-ipadic-2.7.0-20070801-neologd-20200910/ /LDAdash/mecab-ipadic-neologd

RUN apt update && apt install -y mecab \
libmecab-dev \
mecab-ipadic-utf8 \
git \
make \
curl \
xz-utils \
file \
swig \
sudo && \
pip install -r requirements.txt


#ENV MECABPATH=/LDAdash/mecab-ipadic-neologd

EXPOSE 80 2222

CMD ["python","LDAdash.py"]