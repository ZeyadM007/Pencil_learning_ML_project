from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow_hub as hub
import json
import numpy as np

model = load_model('final_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
with open('final_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def predict_sentence(word):
    for _ in range(100):
        input_for_prediction = np.asarray([word])
        predicted = model.predict([input_for_prediction, input_for_prediction], verbose=0)
        predicted = np.argmax(predicted, axis=1)
        output_word = ''
        for id_, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = id_
                break
        word += ' ' + output_word
    return word

print(predict_sentence('What authority surfeits on would'))