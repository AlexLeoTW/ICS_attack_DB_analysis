# Dataset: Industrial Control System (ICS) Cyber Attack Datasets
#            / Dataset 1: Power System Datasets
# https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

FILE_PATH = '../../2_classes/data1.csv'
STEP_SIZE = 20
BATCH_SIZE = 5
# shape = BATCH_SIZE, STEP_SIZE, columns

import os, random, math
import numpy as np
import pandas as pd
import encoder as enc
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# read file
print('===== read file =====')
dirpath = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.abspath(os.path.join(dirpath, FILE_PATH))
df = pd.read_csv(filepath)
print(df.info())

# dealing with: NaN, ∞, -∞
print('===== cleanup =====')
def cleanup(df):
    def remove(df):
        df = df[df.isin([np.nan, np.inf, -np.inf]).any(axis=1) == False]

    def replace_with_high_std(df, std_increase=10):
        df.dropna(axis=1, how='any', inplace=True)

        cols_name_w_missing = df.columns[df.isin([np.nan, np.inf, -np.inf]).any(axis=0)]
        cols_w_missing = df[cols_name_w_missing]
        # print(cols_w_missing)
        wo_missing = cols_w_missing[cols_w_missing.abs() < np.inf]
        cols_max_std = (wo_missing - wo_missing.mean()).abs().max() / wo_missing.std()

        new_inf_val = wo_missing.mean() + wo_missing.std() * (cols_max_std + std_increase)
        new_neg_inf = wo_missing.mean() - wo_missing.std() * (cols_max_std + std_increase)

        df.replace(np.inf, new_inf_val, inplace=True)
        df.replace(-np.inf, new_neg_inf, inplace=True)

    def replace_with_type_min_max(df):
        df.dropna(axis=1, how='any', inplace=True)

        for col in df.columns:
            col_type = df[col].dtype
            print('preprocessing {}({})...'.format(col, col_type))
            if col_type in ['float16', 'float32', 'float64']:
                df[col].replace(np.inf, np.finfo(col_type).max, inplace=True)
                df[col].replace(-np.inf, np.finfo(col_type).min, inplace=True)
            if col_type in ['int16', 'int32', 'int64']:
                df[col].replace(np.inf, np.iinfo(col_type).max, inplace=True)
                df[col].replace(-np.inf, np.iinfo(col_type).min, inplace=True)

    remove(df)
    # replace_with_high_std(df)
    # replace_with_type_min_max(df)
cleanup(df)

# encode
print('===== encode =====')
def encode(df):
    colums = enc.encode_numeric_zscore(df)
    print('Z-scored colums: \n  {}'.format(colums))
    classes = enc.encode_text_index(df, 'marker')
    print('marker classes: \n  {}'.format(classes))
encode(df)
df.dropna(inplace=True,axis=1)

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
print('===== to_xy =====')
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].astype(np.float32), dummies.astype(np.float32)
    else:
        # Regression
        return df[result].astype(np.float32), df[target].astype(np.float32)

x, y = to_xy(df,'marker')
print('x.shape: {}, {}'.format(x.shape, type(x)))
print('y.shape: {}, {}'.format(y.shape, type(y)))

# Create time sequences of x, y with time_step
print('===== create sequences =====')
def to_sequences(x, y, step_size=STEP_SIZE):
    if x.shape[0] != y.shape[0]:
        return None, None

    step_count = x.shape[0] - step_size + 1
    x_seq = np.zeros((step_count, step_size, x.shape[1]))
    y_seq = np.zeros((step_count, step_size, y.shape[1]))
    for i in range(step_count):
        x_seq[i] = x[i:i+step_size]
        y_seq[i] = y[i:i+step_size]

    return x_seq, y_seq

x_seq, y_seq = to_sequences(x, y, step_size=STEP_SIZE)

# Create a test/train split.  20% test
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq,
    test_size=0.2, random_state=42)
print('step_size = {}'.format(STEP_SIZE))

# RNN stuff
print('===== setup RNN =====')
# for x_train, y_train in train_gen:
    # print(y_train)
#

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(STEP_SIZE, x.shape[1]),
    return_sequences = True))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,
    validation_data=(x_test,y_test),
    callbacks=[monitor],
    verbose=1, epochs=200)
model.summary()
