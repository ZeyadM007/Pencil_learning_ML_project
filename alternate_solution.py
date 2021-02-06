import tensorflow_hub as hub
import tensorflow as tf
from tensorflow_hub import KerasLayer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from csv import reader

file_path = get_file('shakespear2.txt', 'https://homl.info/shakespeare')

with open(file_path) as f:
    data = f.read()

#Spliting the lines of the text into a list
corpus = data.lower().split('\n')

#Tokenizer used to assign a value to each word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []

#Tokenizer used to convert the texts into sequences using the values the tokenizer assigned to each word
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

#Taking the last word to make it the target for training
xs = []
ys = []
for line in input_sequences:
    xs.append(line[:-1])
    ys.append(line[-1])

#Padded the input_sequences to make them all uniform. Then took all the columns of the training except the last one.
max_sequence_len = max([len(x) for x in input_sequences])
alternate_training = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
alternate_training = alternate_training[:,:-1]
ys = np.array(ys)

#This model is a Sequential model unlike the main_model. This model uses custom made Embeddings to create a 64 dimensional vector. The similarity in both models is
# they both run with sparse_categorical loss function.
model_alt = Sequential()
model_alt.add(Embedding(12633, 64, input_length=max_sequence_len-1))
model_alt.add(Conv1D(128, 5, activation='relu'))
model_alt.add(GlobalAveragePooling1D())
#model_alt.add((LSTM(20)))
model_alt.add(Dense(128, activation='relu'))
model_alt.add(Dense(12633, activation='softmax'))

model_alt.summary()

model_alt.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_alt.fit(alternate_training, ys, epochs=2, verbose=1)

def predict_sentence_alt(word):
    for _ in range(100):
        token_list = tokenizer.texts_to_sequences([word])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model_alt.predict_classes(token_list, verbose=0)
        output_word = ''
        for id_, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = id_
                break
        word += ' ' + output_word
    return word

print(predict_sentence_alt('What authority surfeits on would'))
