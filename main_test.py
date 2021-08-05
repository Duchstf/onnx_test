import numpy as np
import random as rnd
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
import sys
import onnx
import keras2onnx
import hls4ml
import pytest

np.set_printoptions(threshold=sys.maxsize)

rnd.seed(42)
np.random.seed(42)

height = 10
width = 10
chan = 3

input_shape = (height,width,chan)
input_shape1d = (width,chan)
num_classes = 3

sparcity = 0.
int_bits = 6

# Hint for testing: try commenting out specific layers and generating new models that way

def nonsense_2d_model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='linear', input_shape=input_shape))
    model.add(BatchNormalization())
    #model.add(ReLU()) # Note that this doesn't end up being Relu but Clip???
    model.add(SeparableConv2D(8, kernel_size=(5, 5), padding='valid', strides=(1, 1), activation='linear', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(PReLU())
    #model.add(DepthwiseConv2D(kernel_size=(2, 2), padding='valid', strides=(1, 1), activation='linear'))
    model.add(Activation(activation='selu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    #model.add(Dense(10, activation='tanh'))
    model.add(Dense(num_classes, activation='sigmoid'))

    # It really makes no sense to add more layers, but we should try putting one of these above
    #model.add(GlobalAveragePooling2D())
    #model.add(UpSampling2D(size=(2,1), input_shape=input_shape))

    return model

def equally_dumb_1d_model():
    model = Sequential()
    model.add(Conv1D(16, 4, padding='same', activation='relu', use_bias=True, input_shape=input_shape1d))
    model.add(Dense(5, activation='relu')) # Try not to trip on this ;-)
    model.add(Conv1D(4, 1, padding='same', activation='selu', use_bias=False))
    #model.add(MaxPooling1D(pool_size=2)) # This one breaks keras2onnx, WTF
    model.add(Conv1D(4, 1, padding='valid', activation='linear', use_bias=False))
    #model.add(GlobalAveragePooling1D())
    model.add(Softmax())

    return model

def a_functional_model_with_branches():
    # This is in fact an initial block of ENet
    inputs = Input(shape=input_shape)
    conv = Conv2D(5, kernel_size=3, use_bias=False, padding='valid')
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
    pool_pad = ZeroPadding2D(padding=((0, 2), (0, 2)))
    bn = BatchNormalization()
    concat = Concatenate(axis=-1)
    act = Activation('relu')

    pool = pool(inputs)
    x = pool_pad(pool)
    x = conv(x)
    x = concat([x, pool])
    x = bn(x)
    x = act(x) #some how this doesn't appear in model graph when converted to onnx

    model = Model(inputs, x)
    model.compile()
    return model

def set_some_weights(model):
    for layer in model.layers:
         old_weights = layer.get_weights()
         if len(old_weights) > 0:
              new_weights = []
              for w in old_weights:
                   print(layer.name, w.shape)
                   n_zeros = 0
                   if sparcity > 0:
                        n_zeros = int(sparcity * np.prod(w.shape))
                   if n_zeros > 0:
                        zero_indices = rnd.sample(range(1, np.prod(w.shape)), n_zeros)
                   else:
                        zero_indices = []
                   new_w = []
                   for i in range(np.prod(w.shape)):
                        if i in zero_indices:
                             new_w.append(0)
                        else:
                             #new_w.append(rnd.randint(1, 2**(int_bits - 1)))
                             #new_w.append(rnd.randint(1, 3))
                             new_w.append(rnd.uniform(0, 0.5))
                   new_w = np.asarray(new_w).reshape(w.shape)
                   new_weights.append(new_w)
              layer.set_weights(new_weights)

def create_predict_data(model):
    x = np.random.rand(np.prod(model.input.shape[1:])).reshape(model.input.shape[1:])
    x = np.expand_dims(x, axis=0)

    return x

def predict(model, x, print_result=True):
    predictions = model.predict(x)
    if print_result:
        print(predictions.flatten())

    return predictions

def convert_to_onnx(model, filename=None):
    onnx_model = keras2onnx.convert_keras(model)
    if filename is not None:
        onnx.save(onnx_model, filename)

    return onnx_model

def convert_to_hls4ml(onnx_model, output_dir='output_hls'):
    hls_model = hls4ml.converters.convert_from_onnx_model(onnx_model, output_dir=output_dir)
    hls_model.compile()

    return hls_model

def compare_predictions(keras_predictions, hls_predictions):
    # TODO Compare Keras and hls4ml predictions
    pass

#######################
# Now the actual tests
#######################

def test_1d():
    model = equally_dumb_1d_model()
    set_some_weights(model)
    model.summary()

    onnx_model = convert_to_onnx(model, filename='onnx_1d.onnx')
    hls_model = convert_to_hls4ml(onnx_model, output_dir='hls_1d')

    # x = create_predict_data(model)
    # keras_pred = predict(model, x, print_output=True)
    # hls_pred = predict(hls_model, x, print_output=True)

    # compare_predictions(keras_pred, hls_pred)

def test_2d():
    model = nonsense_2d_model()
    set_some_weights(model)
    model.summary()

    onnx_model = convert_to_onnx(model, filename='onnx_2d.onnx')
    hls_model = convert_to_hls4ml(onnx_model, output_dir='hls_2d')

    #x = create_predict_data(model)
    #keras_pred = predict(model, x, print_output=True)
    #hls_pred = predict(hls_model, x, print_output=True)

    #compare_predictions(keras_pred, hls_pred)

def test_branch():
    model = a_functional_model_with_branches()
    set_some_weights(model)
    model.summary()

    onnx_model = convert_to_onnx(model, filename='onnx_enet.onnx')

    hls_model = convert_to_hls4ml(onnx_model, output_dir='hls_enet')

    # x = create_predict_data(model)
    # keras_pred = predict(model, x, print_output=True)
    # hls_pred = predict(hls_model, x, print_output=True)

    # compare_predictions(keras_pred, hls_pred)

if __name__ == '__main__':
    #pytest.main([__file__])
    test_branch()
    # Or just call test_1d() and test_2d()