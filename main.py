import tensorflow as tf

from typing import Union

from fastapi import FastAPI

import numpy as np

model = tf.keras.models.load_model('model')
app = FastAPI()

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
    return {'Did it work?' : 'Yes it did!'}
