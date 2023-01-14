FROM python:3.10-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip --default-timeout=1000 install -r requirements.txt
COPY . .

EXPOSE 8000

CMD python -m uvicorn api.main:app --host 0.0.0.0 --port 8000