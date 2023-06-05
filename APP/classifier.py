"""Model definition for Spam Classification
A sequential model for text classification.

Model Architecture:
- Embedding layer with `max_words` vocabulary size, `EMBED_DIM` embedding dimension, and input length determined by the shape of X_train.
- SpatialDropout1D layer with a dropout rate of 0.4.
- LSTM layer with `LSTM_OUT` units, dropout rate of 0.3, and recurrent dropout rate of 0.3.
- Dense layer with 2 units and softmax activation for multi-class classification.

Compilation Configuration:
- Loss function: categorical cross-entropy.
- Optimizer: Adam.
- Evaluation metric: accuracy.

Note: The softmax activation function returns probabilities for each class.
"""
import json
import pickle
import pathlib
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, SpatialDropout1D
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences


BASE_DIR = pathlib.Path().resolve().parent
EXPORT_DIR = BASE_DIR / "exports"

SPAM_DATASET_PATH = EXPORT_DIR / "spam-dataset.csv"
METADATA_EXPORT_PATH = EXPORT_DIR / 'spam-metadata.pkl'
TOKENIZER_EXPORT_PATH = EXPORT_DIR / 'spam-tokenizer.json'
MODEL_EXPORT_PATH = EXPORT_DIR / 'spam-model.h5'

EMBED_DIM = 128
LSTM_OUT = 196
BATCH_SIZE = 32
EPOCHS = 5

with open(TOKENIZER_EXPORT_PATH, 'r') as f:
    data = json.load(f)
    tokens = json.dumps(data)
    tokenizer = tokenizer_from_json(tokens)

df = pd.read_csv(SPAM_DATASET_PATH, index_col=[0])
df.head()

with open(METADATA_EXPORT_PATH, 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
X_train = data['X_train']
y_test = data['y_test']
y_train = data['y_train']
legend = data['label_legend']
max_sequence = data['max_seq_length']
max_words = data['max_words']

model = Sequential()

model.add(Embedding(max_words, EMBED_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(LSTM_OUT, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(2, activation='softmax')) # remenber: SoftMax return is on % 

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS)
model.save(str(MODEL_EXPORT_PATH))

def predict(text_str:str, max_words:int = 280, max_sequence:int = 280, tokenizer:Tokenizer = None):
    """Predicts class of string input

    Args:
        text_str (str): string input to classification
        max_words (int, optional): max number of words acepted. Defaults to 280.
        max_sequence (int, optional): max sequece of digits acepted. Defaults to 280.
        tokenizer (_type_, optional): tokenizer object. Defaults to None. If None return None

    Returns:
        Dict: probability description of classification result
        None: case of no Tokeniner defined
    """
    if not tokenizer:
        return None

    sequences = tokenizer.texts_to_sequences([text_str])

    x_input = pad_sequences(sequences, maxlen=max_sequence)
    y_output = model.predict(x_input)

    top_y_index = np.argmax(y_output)
    preds = y_output[top_y_index]

    result = {'han': preds[0], 'spam': preds[1]}
    return result

predict("Hello world", max_words=max_words, max_sequence=max_sequence, tokenizer=tokenizer)
