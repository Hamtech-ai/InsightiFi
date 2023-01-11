from fastapi import FastAPI
from runAll import runModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"HamTech": ["Alireza", "Masoud", "Mohammad", "Shabnam"]}

@app.get("/get_predict")
def get_predict():
    resp = runModel()
    return resp