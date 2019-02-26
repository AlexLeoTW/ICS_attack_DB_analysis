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

# Create a test/train split.  20% test
def train_valid_gen_factory(x, y, batch_size=BATCH_SIZE, step_size=STEP_SIZE, validation_split=0.2):
    if x.shape[0] != y.shape[0]:
        return None, None

    step_count = x.shape[0] - step_size + 1
    random_indexes = random.sample(range(0, step_count), k=step_count)        # shuffle ( 0 ~ step_count )
    # random_indexes = list(range(0, step_count))
    slice_index = math.ceil(len(random_indexes) * validation_split)
    valid_indexes = random_indexes[ :slice_index]
    train_indexes = random_indexes[slice_index: ]
    valid_size = len(valid_indexes)
    train_size = len(train_indexes)

    def gen_batch(x, y, index_list, batch_size, step_size):
        while True:
            random.shuffle(index_list)
            x_batch = np.zeros((batch_size, step_size, x.shape[1]))
            y_batch = np.zeros((batch_size, step_size, y.shape[1]))

            while len(index_list) > 0:
                local_index_list = list(index_list)
                start_index = local_index_list.pop()

                for count in range(0, min(len(local_index_list), batch_size)):
                    x_batch[count] = x[start_index : start_index + step_size]
                    y_batch[count] = y[start_index : start_index + step_size]
                    # print(y_batch)

                yield x_batch, y_batch

    valid_gen = gen_batch(x, y, valid_indexes, batch_size, step_size)
    train_gen = gen_batch(x, y, train_indexes, batch_size, step_size)

    return train_gen, valid_gen, train_size, valid_size

print('batch_size = {}'.format(BATCH_SIZE))
print('step_size = {}'.format(STEP_SIZE))
train_gen, valid_gen, train_size, valid_size = train_valid_gen_factory(x, y,
    batch_size = BATCH_SIZE,
    step_size = STEP_SIZE,
    validation_split = 0.1
)

# RNN stuff
print('===== setup RNN =====')
# for x_train, y_train in train_gen:
    # print(y_train)
#

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(STEP_SIZE, x.shape[1]), return_sequences = True))
model.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(
    train_gen,
    validation_data = valid_gen,
    epochs = 20,
    steps_per_epoch = train_size,
    validation_steps = valid_size,
    callbacks=[monitor]
)
model.summary()
