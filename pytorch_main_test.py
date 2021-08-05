import numpy as np
import random as rnd
import sys
import onnx
import hls4ml
import pytest

import torch.nn as nn
import torch
import torch.nn.functional as F

np.set_printoptions(threshold=sys.maxsize)

rnd.seed(42)
np.random.seed(42)

sparcity = 0.
int_bits = 6

#######################
# Model definition
#######################
# Hint for testing: try commenting out specific layers and generating new models that way
class nonsense_2d_model(nn.Module):   
    def __init__(self):
        super(nonsense_2d_model, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class equally_dumb_1d_model(nn.Module):   
    def __init__(self):
        super(equally_dumb_1d_model, self).__init__()
        self.conv1 = nn.Conv1d(32, 16, 4)
        self.conv2 = nn.Conv1d(16, 5, 1)
        self.fc1 = nn.Linear(5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 4)
        # If the size is a square, you can specify with a single number
        x = F.max_pool1d(F.tanh(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

class a_functional_model_with_branches(nn.Module):
    def __init__(self):
        super(a_functional_model_with_branches, self).__init__()
        self.conv = nn.Conv2d(5, 5, 3, padding = (1,2))
        self.bn = nn.BatchNorm2d(5)

    def forward(self, x):
        pool = F.max_pool2d(x, (2, 2), stride=2)
        x = self.conv(pool)
        x = torch.cat([x, pool], axis = -1)
        x = self.bn(x)
        x = F.relu(x)
        return x


#######################
# Helpers
#######################
def create_predict_data(model):
    x = np.random.rand(np.prod(model.input.shape[1:])).reshape(model.input.shape[1:])
    x = np.expand_dims(x, axis=0)

    return x

def predict(model, x, print_result=True):
    predictions = model.predict(x)
    if print_result:
        print(predictions.flatten())

    return predictions

def convert_to_onnx(model, filename=None, input_dims = [1, 1, 32, 32]):

    dummy_input = torch.randn(input_dims)
    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    torch.onnx.export(model,
                      dummy_input,
                      filename,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=dynamic_axes)

    onnx_model = onnx.load(filename)

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
    #set_some_weights(model)

    onnx_model = convert_to_onnx(model, filename='pytorch_onnx_1d.onnx', input_dims = [1, 32, 16])
    hls_model = convert_to_hls4ml(onnx_model, output_dir='pytorch_hls_1d')

    # x = create_predict_data(model)
    # keras_pred = predict(model, x, print_output=True)
    # hls_pred = predict(hls_model, x, print_output=True)

    # compare_predictions(keras_pred, hls_pred)

def test_2d():
    model = nonsense_2d_model()
    #set_some_weights(model)

    onnx_model = convert_to_onnx(model, filename='pytorch_onnx_2d.onnx', input_dims = [1, 1, 32, 32])
    hls_model = convert_to_hls4ml(onnx_model, output_dir='pytorch_hls_2d')

    #x = create_predict_data(model)
    #keras_pred = predict(model, x, print_output=True)
    #hls_pred = predict(hls_model, x, print_output=True)

    #compare_predictions(keras_pred, hls_pred)

def test_branch():
    model = a_functional_model_with_branches()
    #set_some_weights(model)

    onnx_model = convert_to_onnx(model, filename='pytorch_onnx_enet.onnx', input_dims = [3, 5, 10, 10])
    hls_model = convert_to_hls4ml(onnx_model, output_dir='pytorch_hls_enet')

    # x = create_predict_data(model)
    # keras_pred = predict(model, x, print_output=True)
    # hls_pred = predict(hls_model, x, print_output=True)

    # compare_predictions(keras_pred, hls_pred)

if __name__ == '__main__':
    #pytest.main([__file__])
    # Or just call test_1d() and test_2d()

    #test_1d()
    #test_2d()
    test_branch()