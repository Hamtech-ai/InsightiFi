docker volume create shared_data
docker build -t prediction:latest ./prediction
docker build -t fast-api:latest ./api
docker-compose up --build