# About this project
This is a tool for traders to help them decide to buy/sell/hold, or for ML engineers who want to make their hands dirty in working with Time series/Economical data. 

# How to Use
1. Clone the project:
```command
git clone https://github.com/Hamtech-ai/iran-stock-market
```
2. Build and run application in the background.
```command
docker-compose up --build -d
```
  - up will start the whole application.
  - --build will first build the application before starting it.
  - -d will run the application in the background.

# Wiki Page
To check complete Documentation for all steps check our [wiki page](https://github.com/Hamtech-ai/iran-stock-market/wiki).

# Dataset
The following three primary reasons led us to choose [pytse-client](https://github.com/Glyphack/pytse-client) library as our ML dataset of the Iranian stock market:
1. Data is updated daily and non-stop.
2. Contains valuable data about individual sales and purchases.
3. Has a history of buying stocks over long periods.

# TODO
- [ ] Using a cron job to run the model daily.
- [ ] Deployment With Docker Containers.
- [ ] Using NLP tools to enhance accuracy of model in daily predictions.

# Contribute
Before opening a [PR](https://github.com/Hamtech-ai/iran-stock-market/pulls), please read our [contributor](/.github/CONTRIBUTING.md) guide. This project exists thanks to all the people who contribute:
<p align="center"><a href="./graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Hamtech-ai/iran-stock-market" />
</a></p>