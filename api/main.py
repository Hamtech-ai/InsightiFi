import pandas as pd
from fastapi import FastAPI

df = pd.read_csv('./inputs/output_for_API.csv')

description = """
## Root
Hamtech members!
## Get Predict
Buy or sell probability in a daily timeframe (a closer probability to one indicates a stronger buy position, while a closer probability to zero indicates a stronger sell position). 🚀
"""
app = FastAPI(
    title = "buy/sell signals for Mobarakeh steel company shares",
    version = "0.0.1",
    description = description,
    docs_url="/docs"
)

@app.get("/")
def read_root():
    return {"HamTech": ["Alireza", "Masoud", "Mohammad", "Shabnam"]}

@app.get("/get_predict")
def get_predict():
    resp = df.set_index('jdate')['labelProb'].to_dict()
    return resp