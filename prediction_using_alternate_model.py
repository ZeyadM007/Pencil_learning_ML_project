from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

model = load_model('alternate_model.h5')
with open('final_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def predict_sentence_alt(word):
    for _ in range(100):
        token_list = tokenizer.texts_to_sequences([word])[0]
        token_list = pad_sequences([token_list], maxlen=15, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ''
        for id_, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = id_
                break
        word += ' ' + output_word
    return word

print(predict_sentence_alt('What authority surfeits on would'))