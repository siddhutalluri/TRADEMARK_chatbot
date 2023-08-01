import uvicorn
from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = FastAPI()

def load_model_and_tokenizer():
    model = tf.keras.models.load_model("trademarker.h5")
    with open('tokenizer.json') as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
    with open('label_encoder.json') as f:
        label_encoder_json = f.read()
        label_encoder = json.loads(label_encoder_json)
    return model, tokenizer, label_encoder

def preprocess_input(input_text, tokenizer, max_length):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    return padded_sequence

def predict_class(input_text):
    model, tokenizer, label_encoder = load_model_and_tokenizer()
    max_length = model.input_shape[1]
    preprocessed_input = preprocess_input(input_text, tokenizer, max_length)
    predicted_probabilities = model.predict(preprocessed_input)
    predicted_class_index = tf.argmax(predicted_probabilities, axis=1).numpy()[0]
    predicted_class = label_encoder[predicted_class_index]
    return predicted_class

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict'/<'string:inp_txt'>)
def predict_cl(data: inp_txt):
    data = data.dict()
    text = data['text']
    predicted_class = predict_class(text)
    return {'prediction': predicted_class}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
