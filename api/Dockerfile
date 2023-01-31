FROM python:3.10.6-slim-buster

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt
RUN pip --no-cache-dir install -r requirements.txt
COPY . .

EXPOSE 8000

# CMD [ "ls", "/src/inputs/" ]
CMD ["uvicorn", "main:app","--reload", "--host", "0.0.0.0", "--port", "8000"]
# docker run -p 8000:8000 fast-api:latest