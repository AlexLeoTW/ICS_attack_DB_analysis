# Dataset: Industrial Control System (ICS) Cyber Attack Datasets
#            / Dataset 1: Power System Datasets
# https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

ANN_NAME = 'SimpleRNN'
VALIDATION_SPLIT = 0.2

import os, sys, time
import numpy as np
import pandas as pd
import encoder as enc
import common as common
import cmdargv as cmdargv
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import metrics

options = cmdargv.parse_argv(sys.argv, ANN_NAME)

# read file
print('===== read file =====')
df = pd.read_csv(options.dataset)
print(df.info())
common.dropp_columns_regex(df, options.exclude)

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
print('step_size = {}'.format(options.step_size))
x_seq = common.to_sequences(x, step_size=options.step_size)
y_seq = y[options.step_size-1:]
print('x_seq.shape: {}, {}'.format(x_seq.shape, type(x_seq)))
print('y_seq.shape: {}, {}'.format(y_seq.shape, type(y_seq)))

# Create a test/train split.  20% test
print('validation_split = {}%'.format(VALIDATION_SPLIT * 100))
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq,
    test_size=VALIDATION_SPLIT, random_state=42)

# SimpleRNN stuff
print('===== setup SimpleRNN =====')
callback_monitor='val_acc'
earlyStopping = EarlyStopping(monitor=callback_monitor, min_delta=1e-3, patience=20, mode='auto', verbose=1)
modelCheckpoint = ModelCheckpoint(filepath=options.model_path, monitor=callback_monitor, mode='auto', save_best_only=True, verbose=1)
common.dir_create_if_not_exist(options.model_path)

model = Sequential()
model.add(SimpleRNN(units=options.units, input_shape=(options.step_size, x_train.shape[2])))
model.add(Dense(units=y_train.shape[1], kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()    # -------------------------------------------------┐
result = model.fit(x_train, y_train, #                                         │
    validation_data=(x_test, y_test), #              train_time                │
    callbacks=[earlyStopping, modelCheckpoint],    #                           │
    batch_size=128,         #                                                  |
    verbose=1, epochs=300)  #                                                  │
train_time = time.time() - start_time   # -------------------------------------┘
model.summary()

print('train_time = {}s'.format(train_time))

# measure accuracy
print('===== measure accuracy =====')
model = load_model(options.model_path)
start_time = time.time()    # -------------------------------------------------┐
pred = model.predict(x_test) #                       val_time                  |
val_time = time.time() - start_time   # ---------------------------------------┘
pred = np.argmax(pred, axis=1)
y_eval = np.argmax(y_test.values, axis=1)

acc_score = metrics.accuracy_score(y_eval, pred)
print('Accuracy =', acc_score*100)

reca_score = metrics.recall_score(y_eval, pred)
print('Recall =', reca_score*100)

prec_score = metrics.precision_score(y_eval, pred)
print('Precision =', prec_score*100)

f1_score = metrics.f1_score(y_eval, pred)
print('F-score =', f1_score)

# write files
print('===== write files =====')
print('saving model "{}"...'.format(os.path.basename(options.model_path)))
common.save_model(model= model, path= options.model_path)

print('saving epochs log "{}"...'.format(os.path.basename(options.log_path)))
common.save_log(fit_result= result, path= options.log_path)

epochs = len(result.history['val_acc'])
best_epoch = result.history['val_acc'].index(max(result.history['val_acc'])) + 1

print('saving history statistics "{}"...'.format(os.path.basename(options.statistics_path)))
common.save_statistics(
    ann_name= ANN_NAME,
    path= options.statistics_path,
    entries= {
        'step_size': options.step_size,
        'units': options.units,

        'acc_score': acc_score,
        'reca_score': reca_score,
        'prec_score': prec_score,
        'f1_score': f1_score,

        'train_time': train_time,
        'epochs': epochs,
        'avg_epoch_time': train_time / epochs,
        'best_epoch': best_epoch,
        'avg_val_time': val_time / epochs,

        'last_val_loss': result.history['val_loss'][-1],
        'last_val_acc': result.history['val_acc'][-1],
        'last_loss': result.history['loss'][-1],
        'last_acc': result.history['acc'][-1],
    },
    drop_duplicates= ['ann_name', 'step_size', 'units'] if options.drop_duplicates else False
)
