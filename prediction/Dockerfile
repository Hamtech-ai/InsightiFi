FROM python:3.10.6-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN apt-get update &&\
    apt-get install -y cron &&\ 
    pip --no-cache-dir install -r requirements.txt
COPY . .

RUN chmod +x /app/run_all.py &&\
    echo "* * * * * python3 /app/run_all.py" >> /var/spool/cron/crontabs/root

CMD cron &&\
    python3 -m run_all --host 0.0.0.0 --port 8888 &&\
    jupyter notebook --ip='*' --port=8888 --no-browser --allow-root