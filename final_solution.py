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

#COnverting the training back to words because the nnlm model and the universal sentence encoder take in words as the input.
training = tokenizer.sequences_to_texts(xs)
training = np.array(training)
ys = np.array(ys)

#Importing universal sentence encoder and nnlm model from TF Hub using tensorflow_hub
universal_sentence_encoder = KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", input_shape=[], output_shape=[512],dtype=tf.string)
nnlm_dim50 = KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2", trainable=False, input_shape=[], output_shape=[50],dtype=tf.string)

model_1 = Sequential()
model_1.add(universal_sentence_encoder)
model_1.add(Dense(32, activation='relu'))

model_2 = Sequential()
model_2.add(nnlm_dim50)
model_2.add(Dense(32, activation='relu'))

merged = Concatenate()([model_1.output, model_2.output])
merged = Dense(32, activation='relu')(merged)
merged = Dense(12633, activation='softmax')(merged)

#Model is a combination of 2 Sequential models each with their respective weights. The models are then combined at merged variable to produce 1 final outcome.
model = Model(inputs=[model_1.input, model_2.input], outputs=[merged])

#Used sparse_categorical_crossentropy as a loss function and used Adam as an optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit([training, training], ys, epochs=10, verbose=1)


#This function outputs the next 100 predicted words
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
