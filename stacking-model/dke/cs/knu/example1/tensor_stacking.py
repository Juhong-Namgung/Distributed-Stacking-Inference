
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
    #nn = Dense(64, input_shape=(32,))(nn)
    #nn = Dense(64, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(64,))(nn)
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
    #nn = Dense(64, input_shape=(64,))(nn)
    nn = Dense(128, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    nn = Dense(64, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(64,))(nn)
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
    #nn = Dense(64, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(64,))(nn)
    nn = Dense(32, input_shape=(64,))(nn)
    main_output = Dense(1, activation='tanh', name='nn3_output')(nn)

    model = Model(input=[main_input], output=[main_output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return model

def     nnsecond():
    main_input = Input(shape=(3,), dtype='float32', name='final_input')
    nn = Dense(8, input_shape=(3,))(main_input)
    nn = Dense(32, input_shape=(8,))(nn)
    #nn = Dense(64, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(64,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(128, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(128,))(nn)
    #nn = Dense(64, input_shape=(64,))(nn)
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
batch_size = 128

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

    #frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

    #tf.train.write_graph(frozen_graph, "./models/", str(i)+"_model.pb", as_text=False)

    #builder = tf.saved_model.builder.SavedModelBuilder("./models/"+str(i) +"NN/")
    #builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.global_variables_initializer())
    #builder.save(False)

    #i = i+1

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
#storm_data = [[[1.0, -1.0, 1.0]]]
#print("FINAL PREDICT")
#print(second_model.predict(storm_data))

'''
print()
print("########################################")
print(x_stack)
print("########################################")
print()
model1_stacker = LinearRegression() # Second-level model
model2_stacker = RidgeCV(alphas=np.logspace(-2, 3))
model3_stacker = LassoCV()

stack_archive = pd.DataFrame(columns=['models', 'prediction', 'score'])

for stacker in [model1_stacker, model2_stacker, model3_stacker]:
    prediction = np.zeros(len(x_stack))
    models = []
    for t, v in KFold(5, random_state=0).split(x_stack):
        x_train = x_stack[t]
        x_val = x_stack[v]
        y_train = y_stack.iloc[t]
        y_val = y_stack.iloc[v]

        stacker.fit(x_train, y_train)
        prediction[v] = stacker.predict(x_val)
        models.append(stacker)
    score = rmse(prediction, y_stack)
    print(score)
    stack_archive = stack_archive.append({'models':models, 'prediction': prediction, 'score': score}, ignore_index=True)

    print(stack_archive.head())

# model coefficient

# linear regression
print(np.mean([model.coef_ for model in stack_archive.iloc[0, 0]], axis=0))

# ridge regression
print(np.mean([model.coef_ for model in stack_archive.iloc[1, 0]], axis=0))

# lasso regression
print(np.mean([model.coef_ for model in stack_archive.iloc[2, 0]], axis=0))

'''




'''
kfold = KFold(n_splits=5, shuffle=True, random_state=33)
for train, validation in kfold.split(x_data, y_data):
    x_train = x_data.iloc[train]
    x_val = x_data.iloc[validation]
    y_train = y_data.iloc[train]
    y_val = y_data.iloc[validation]
    prediction = np.zeros(len(x_data))

    model = nn()
    model.fit({'main_input':x_train}, y_train, epochs=epochs, batch_size=batch_size)

    prediction[validation] = model.predict(x_val)
    score = rmse(np.expm1(prediction), np.expm1(y_data))
    print(score)

    loss, k_accuracy = (model.evaluate(x_val, y_val, verbose=1))
    accuracy.append(k_accuracy)

print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
print('\nK-fold cross validation Accuracy mean: ', np.array(accuracy).mean())
'''
#frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

#tf.train.write_graph(frozen_graph, "../../../models/", "keras_1DCNN.pb", as_text=False)

#builder = tf.saved_model.builder.SavedModelBuilder("../../../models/1DCNN/")
#builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.global_variables_initializer())
#builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.local_variables_initializer())
#builder.save(False)




# # 노드 이름 확인
# f = gfile.FastGFile("./storm/model/CNN/saved_model.pb", 'rb')
# graph_def = tf.GraphDef()
#
# graph_def.ParseFromString(f.read())
# sess = tf.Session()
# sess.graph.as_default()
# tf.import_graph_def(graph_def)

# for op in sess.graph.get_operations():
#     print(op.name)
'''

## Load data URL
DATA_HOME ='../../../data/'
df = pd.read_csv(DATA_HOME + 'urls.csv',encoding='ISO-8859-1', sep=';')

# Initial Data Preparation URL

# Step 1: Convert raw URL string in list of lists where characters that are contained in "printable" are sotred encoded as integer
url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]

# Step 2: Cut URL string at max_len or pad with zeros if shorter
max_len = 75
X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)

# Step 3: Extract labels form df to nupy array
target = np.array(df.label)

print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ' , target.shape)

# Simple Cross-Validation: Split the data set into training and test data
X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=0.2, random_state=33)

# General get layer dimensions for any model
def print_layers_dims(model):
    l_layers = model.layers
    # Note None is ALWAYS batch_size
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ' , l_layers[i].input_shape, 'Output Shpae: ', l_layers[i].output_shape)

# General save model to disk function
def save_model(fileModelJSON, fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON, 'w') as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)

# General load model from disk function
def load_model(fileModelJSON, fileWeights):
    with open(fileModelJSON, 'r') as f:
        model_json = json.load(f)
        model = model_from_json(model_json)

    model.load_weights(fileWeights)
    return model

# 1D Convolution and Fully Connected Layers
def conv_fully(max_len=75, emb_dim=32, max_vocab_len=100, W_reg=regularizers.l2(1e-4)):
    # Input
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

    # Embedding layer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(main_input)
    emb = Dropout(0.25)(emb)

    def sum_1d(X):
        return K.sum(X, axis=1)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        # Conv layer
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
        conv = ELU()(conv)
        conv = Lambda(sum_1d, output_shape=(filters,))(conv)
        conv = Dropout(0.5)(conv)

        return conv

    # Multiple Conv Layers
    conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

    # Fully Connected Layers
    merged = concatenate([conv1, conv2], axis=1)

    hidden1 = Dense(1024)(merged)
    hidden1 = ELU()(hidden1)
    hidden1 = BatchNormalization(mode=0)(hidden1)
    hidden1 = Dropout(0.5)(hidden1)

    hidden2 = Dense(1024)(hidden1)
    hidden2 = ELU()(hidden2)
    hidden2 = BatchNormalization(mode=0)(hidden2)
    hidden2 = Dropout(0.5)(hidden2)

    # Output layer (last fully connected layer)
    output = Dense(1, activation='sigmoid', name='main_output')(hidden2)

    # Compile model and define optimizer
    model = Model(input=[main_input], output=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

def save_model(session, input_tensor, output_tensor):
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'input': tf.saved_model.utils.build_tensor_info(input_tensor)},
        outputs = {'output': tf.saved_model.utils.build_tensor_info(output_tensor)},
    )
    b = tf.saved_model.builder.SavedModelBuilder('../output/models')
    b.add_meta_graph_and_variables(session,
                                   [tf.saved_model.tag_constants.SERVING],
                                   signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    b.save()

# Fit model and Cross-Validation, 4 CONV + Fully Connected
epochs = 5
batch_size = 64

# Use 5-fold cross validation
accuracy = []
kfold = KFold(n_splits=5, shuffle=True, random_state=33)
for train, validation in kfold.split(X, target):
    model = conv_fully()
    model.fit({'main_input':X[train]}, target[train], epochs=epochs, batch_size=batch_size)

    loss, k_accuracy = (model.evaluate(X[validation], target[validation],verbose=1))
    accuracy.append(k_accuracy)

print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
print('\nK-fold cross validation Accuracy mean: ', np.array(accuracy).mean())

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "../../../models/", "keras_1DCNN.pb", as_text=False)

builder = tf.saved_model.builder.SavedModelBuilder("../../../models/1DCNN/")
builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.global_variables_initializer())
#builder.add_meta_graph_and_variables(K.get_session(),[tf.saved_model.tag_constants.SERVING],main_op=tf.local_variables_initializer())
builder.save(False)

# # 노드 이름 확인
# f = gfile.FastGFile("./storm/model/CNN/saved_model.pb", 'rb')
# graph_def = tf.GraphDef()
#
# graph_def.ParseFromString(f.read())
# sess = tf.Session()
# sess.graph.as_default()
# tf.import_graph_def(graph_def)

# for op in sess.graph.get_operations():
#     print(op.name)
'''
'''
sess = tf.Session()
#sess = tf.InteractiveSession()
K.set_session(sess)
sess.run(tf.local_variables_initializer()
)
sess.run(tf.global_variables_initializer())

#writer = tf.summary.FileWriter('./test/result/model/tensorboard/1')
#writer.add_graph(sess.graph)

'''