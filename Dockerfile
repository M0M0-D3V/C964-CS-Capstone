FROM python:3.12.0-slim

WORKDIR /app

ADD . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

EXPOSE 80

CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:5000"]