FROM python:3.10.6-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip --no-cache-dir install -r requirements.txt
COPY . .

CMD ["python", "-m", "run_all", "--host", "0.0.0.0", "--port", "8888"]