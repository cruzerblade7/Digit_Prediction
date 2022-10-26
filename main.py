import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


model = tf.keras.models.load_model('model')
app = FastAPI()

# cors
origins = [
    "https://sankalpmukim.dev",
    "https://kevinsdigits.sankalpmukim.dev",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8000/",
    "http://localhost:8000/analyze"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def make_prediction(data: list[float]):
    d = np.array(data).reshape(1, 500, 500, 1)
    d = tf.image.resize(d, (28, 28))
    prob_dist = model.predict(np.array(d).reshape(1, 28, 28, 1))[0]
    pred = -1
    for i in range(len(prob_dist)):
        if prob_dist[i] == max(prob_dist):
            pred = i
            break
    print('Probabilities: ', prob_dist)
    print('Digit Recognised: ', pred)
    return {'prediction': pred}