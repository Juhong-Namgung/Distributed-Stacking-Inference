
import numpy as np
import pandas as pd
import os
from string import printable
from sklearn import model_selection
import gc
# import gensim
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import KFold
from pathlib import Path
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.python.platform import gfile
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
import json

import warnings
warnings.filterwarnings("ignore")


# read data
train_data = pd.read_csv("./data/train.csv", index_col='id')
test_data = pd.read_csv("./data/test.csv", index_col='id')

# preprocessing
def process_datatime(df):
    df['date'] = pd.to_datetime(df['date'].astype('str').str[:8])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop('date', axis=1)
    return df

train_data = process_datatime(train_data)
test_data = process_datatime(test_data)

x_data = train_data[[col for col in train_data.columns if col != 'price']]
#print(x_data.head())
y_data = np.log1p(train_data['price'])
del train_data; gc.collect();
#print(x_data.shape)
print('Matrix dimensions of X: ', x_data.shape, 'Vector dimension of target: ' , y_data.shape)
# 1D Convolution and Fully Connected Layers
def nn1():
    main_input = Input(shape=(21,), dtype='float32', name='nn1_input')
    nn = Dense(32, input_shape=(21,))(main_input)
    nn = Dense(32, input_shape=(32,))(nn)
    main_output = Dense(1, activation='tanh', name='nn1_output')(nn)

    model = Model(input=[main_input], output=[main_output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return model
def nn2():
    main_input = Input(shape=(21,), dtype='float32', name='nn2_input')
    nn = Dense(32, input_shape=(21,))(main_input)
    nn = Dense(64, input_shape=(32,))(nn)
    nn = Dense(128, input_shape=(64,))(nn)
    nn = Dense(64, input_shape=(128,))(nn)
    nn = Dense(32, input_shape=(64,))(nn)
    main_output = Dense(1, activation='tanh', name='nn2_output')(nn)

    model = Model(input=[main_input], output=[main_output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return model

def nn3():
    main_input = Input(shape=(21,), dtype='float32', name='nn3_input')
    nn = Dense(32, input_shape=(21,))(main_input)
    nn = Dense(64, input_shape=(32,))(nn)
    nn = Dense(32, input_shape=(64,))(nn)
    main_output = Dense(1, activation='tanh', name='nn3_output')(nn)

    model = Model(input=[main_input], output=[main_output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return model

def nnsecond():
    main_input = Input(shape=(3,), dtype='float32', name='final_input')
    nn = Dense(8, input_shape=(3,))(main_input)
    nn = Dense(32, input_shape=(8,))(nn)
    nn = Dense(8, input_shape=(32,))(nn)
    main_output = Dense(1, activation='tanh', name='final_output')(nn)

    model = Model(input=[main_input], output=[main_output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return model

def rmse(pred, true):
    return -np.sqrt(np.mean((pred-true)**2))

def freeze_session(session, keep_var_names=None, output_names="main_output", clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

epochs = 10
batch_size = 64

# Use 5-fold cross validation
accuracy = []
model_1 = nn1()
model_2 = nn2()
model_3 = nn3()

archive = pd.DataFrame(columns=['models', 'prediction', 'score'])
i=1
for model in [model_1, model_2, model_3]:
    prediction = np.zeros(len(x_data))
    models = []
    for t, v in KFold(5, random_state=0).split(x_data):
        x_train = x_data.iloc[t]
        x_val = x_data.iloc[v]
        y_train = y_data.iloc[t]
        y_val = y_data.iloc[v]
        #stacker.fit(x_data,y_data)
        model.fit({'nn'+str(i)+'_input':x_train}, y_train, validation_data=(x_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)

        models.append(model)
        prediction[v] = model.predict(x_val).reshape(3007,)

        #prediction[v] = model.predict(x_val).reshape(3007,)
    score = rmse(np.expm1(prediction), np.expm1(y_data))
    print(score)
    archive = archive.append({'models':models, 'prediction':prediction, 'score':score}, ignore_index=True)
    i = i + 1
print("#########PREDICITON")
print(archive['prediction'])


test_predictions = np.array([np.mean([model.predict(test_data) for model in models], axis=0) for models in archive['models']])
print("#######################결과(1)##########")
print(archive.head())
print("#######################")


mean_stacked_prediction = np.mean([np.expm1(pred) for pred in archive['prediction']], axis=0)
mean_stacked_score = rmse(mean_stacked_prediction, np.expm1(y_data))

print("#######################")
print("평균 값 방법: " + format(mean_stacked_score))
print("#######################")


x_stack = np.array([np.expm1(pred) for pred in archive['prediction']]).transpose()
y_stack = np.expm1(y_data)

second_model = nnsecond()
prediction = np.zeros(len(x_stack))
models = []

stack_archive = pd.DataFrame(columns=['models', 'prediction', 'score'])

for t, v in KFold(5, random_state=0).split(x_stack):
    x_train = x_stack[t]
    x_val = x_stack[v]
    y_train = y_stack.iloc[t]
    y_val = y_stack.iloc[v]
    second_model.fit({'final_input':x_train}, y_train, validation_data=(x_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)
    #second_model.fit(x_train, y_train)
    prediction[v] = second_model.predict(x_val).reshape(3007,)
    models.append(second_model)
    #print(x_train)

#writer = tf.train.SummaryWriter("./logs/nn_logs", K.get_session().graph)
#file_writer = tf.summary.FileWriter("./logs/nn_logs", K.get_session().graph)
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "./models/", "final_model.pb", as_text=False)

builder = tf.saved_model.builder.SavedModelBuilder("./models/finalNN/")
builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.global_variables_initializer())
#builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.local_variables_initializer())
builder.save(False)

score = rmse(prediction, y_stack)
print(score)

stack_archive = stack_archive.append({'models':models, 'prediction': prediction, 'score': score}, ignore_index=True)
print("#######################")
print(stack_archive.head())
print("#######################")
