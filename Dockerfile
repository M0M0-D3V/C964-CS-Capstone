FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
