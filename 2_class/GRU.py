# Dataset: Industrial Control System (ICS) Cyber Attack Datasets
#            / Dataset 1: Power System Datasets
# https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

DIR_PATH = '../../2_classes/'
STEP_SIZE = 20
BATCH_SIZE = 5
VALIDATION_SPLIT = 0.2
# shape = BATCH_SIZE, STEP_SIZE, columns

import os
import numpy as np
import pandas as pd
import encoder as enc
import common as common
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics

# read file
print('===== read file =====')
# dirpath = os.path.abspath(os.path.dirname(__file__))
# dirpath = os.path.abspath(os.path.join(dirpath, DIR_PATH))
# df, files = common.batch_read_csv(dirpath)

dirpath = os.path.abspath(os.path.dirname(__file__))
dirpath = os.path.abspath(os.path.join(dirpath, DIR_PATH))
dirpath = os.path.abspath(os.path.join(dirpath, 'data1.csv'))
df = pd.read_csv(dirpath)
print(df.info())

# dealing with: NaN, ∞, -∞
print('===== cleanup =====')
dropped_columns = common.cleanup(df)
print('dropped_columns: {}'.format(dropped_columns))

# encode
print('===== encode =====')
def encode(df):
    columns = enc.encode_numeric_zscore(df)
    print('Z-scored columns: \n  {}'.format(columns))
    classes = enc.encode_text_index(df, 'marker')
    print('marker classes: \n  {}'.format(classes))
encode(df)

old_columns = df.columns
df.dropna(inplace=True,axis=1)
print('dropped_columns: \n\t{}'.format(list(set(old_columns) - set(df.columns))))

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
print('===== to_xy =====')
x, y = common.to_xy(df,'marker')
print('x.shape: {}, {}'.format(x.shape, type(x)))
print('y.shape: {}, {}'.format(y.shape, type(y)))

# Create time sequences of x, y with time_step
print('===== create sequences =====')
print('step_size = {}'.format(STEP_SIZE))
x_seq, y_seq = common.to_sequences(x, y, step_size=STEP_SIZE)

# Create a test/train split.  20% test
print('validation_split = {}%'.format(VALIDATION_SPLIT * 100))
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq,
    test_size=VALIDATION_SPLIT, random_state=42)

# LSTM stuff
print('===== setup LSTM =====')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model = Sequential()
model.add(GRU(units=50, input_shape=(STEP_SIZE, x.shape[1]),
    return_sequences = True))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

config = model.get_config()
print(config)

model.fit(x_train, y_train,
    validation_data=(x_test,y_test),
    callbacks=[monitor],
    verbose=1, epochs=200)
model.summary()

# config = model.get_config()
# print(config)

pred = model.predict(x_test)
pred = np.argmax(pred,axis=2)[:,-1]
y_eval = np.argmax(y_test,axis=2)[:,-1]

auc_score = metrics.accuracy_score(y_eval, pred)
print('Accuracy =', auc_score*100)

reca_score = metrics.recall_score(y_eval, pred)
print('Recall =', reca_score*100)

prec_score = metrics.precision_score(y_eval, pred)
print('Precision =', prec_score*100)

f1_score = metrics.f1_score(y_eval, pred)
print('F-score =', f1_score)
