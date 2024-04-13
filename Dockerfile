FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt-get install -y chromium-driver

RUN pip install -r requirements.txt

EXPOSE 8000

CMD python3 main.py