# Dataset: Industrial Control System (ICS) Cyber Attack Datasets
#            / Dataset 1: Power System Datasets
# https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

ANN_NAME = 'GRU'
VALIDATION_SPLIT = 0.2

import os, sys, time
import pandas as pd
import encoder as enc
import common as common
import cmdargv as cmdargv
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

options = cmdargv.parse_argv(sys.argv, ANN_NAME)

# read file
print('===== read file =====')
df = pd.read_csv(options['dataset_path'])
print(df.info())
# common.dropp_columns_regex(df, options['exclude'])

# dealing with: NaN, ∞, -∞
print('===== cleanup =====')
dropped_columns = common.cleanup(df)
print('dropped_columns: {}'.format(dropped_columns))

# encode
print('===== encode =====')
def encode(df):
    columns = enc.encode_numeric_zscore(df, exclude='marker')
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
print('step_size = {}'.format(options['step_size']))
x_seq, y_seq = common.to_sequences(x, y, step_size=options['step_size'])

# Create a test/train split.  20% test
print('validation_split = {}%'.format(VALIDATION_SPLIT * 100))
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq,
    test_size=VALIDATION_SPLIT, random_state=42)

# GRU stuff
print('===== setup GRU =====')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model = Sequential()
model.add(GRU(units=options['units'], input_shape=(options['step_size'], x.shape[1]),
    return_sequences = True))
model.add(Dense(units=y.shape[1], kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()    # -------------------------------------------------┐
result = model.fit(x_train, y_train, #                                         │
    validation_data=(x_test,y_test), #              train_time                 │
    callbacks=[monitor],    #                                                  │
    verbose=1, epochs=300)  #                                                  │
train_time = time.time() - start_time   # -------------------------------------┘
model.summary()

print('train_time = {}s'.format(train_time))

# write files
print('===== write files =====')
common.save_results(
    ANN_NAME,
    options=options,
    model=model,
    validation_data=(x_test, y_test),
    fit_result=result,
    train_time=train_time
)
