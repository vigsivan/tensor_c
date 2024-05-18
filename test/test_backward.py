import pytest
import torch
import ctypes
import numpy as np

@pytest.fixture()
def net_lin_sig():
    class BasicNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = torch.nn.Sigmoid()
            self.linear = torch.nn.Linear(5, 1)

        def forward(self, x):
            return self.act(self.linear(x))

    net = BasicNetwork().train().requires_grad_(True)
    breakpoint()
    input_arr = torch.linspace(1, 5, 5)
    output_arr = net(input_arr)
    loss = torch.sum((torch.Tensor([0]) - output_arr)**2)
    loss.backward()
    yield net

@pytest.fixture()
def net_bias():
    class BiasLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            bias_value = torch.Tensor([6,7,8,9,10])
            self.bias_layer = torch.nn.Parameter(bias_value)
        
        def forward(self, x):
            return x + self.bias_layer

    return BiasLayer()


def net_sse():
    class BiasLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            bias_value = torch.randn((5))
            self.bias_layer = torch.nn.Parameter(bias_value)
        
        def forward(self, x):
            return x + self.bias_layer

    class SSE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = BiasLayer()

        def forward(self, x1, x2):
            return torch.sum((self.bias(x1)-x2)**2)


def test_bias(tlib, net_bias):
    input_arr = torch.linspace(1, 5, 5)
    output_arr = net_bias(input_arr)
    loss = torch.sum(output_arr)
    loss.backward()

    InputShape = ctypes.c_int * 1
    InputData = ctypes.c_float * 5
    input_shape = InputShape(5)
    input_data = InputData(1., 2., 3., 4., 5.)
    input_tensor = tlib.init_tensor(1,input_shape, input_data)

    SumShape = ctypes.c_int*1
    SumData = ctypes.c_float * 5
    sum_shape = SumShape(5)
    sum_data = SumData(6.,7.,8.,9.,10.,)
    sum_tensor = tlib.init_tensor(1,sum_shape, sum_data)

    summation = tlib.op_fp32add(sum_tensor, input_tensor)
    out = tlib.op_fp32total(summation)
    tlib.backward(out)

    # gradient = sum_tensor.contents.gradient
    # expected = [1]*5
    # for i in range(gradient.size):
    #     assert np.allclose(gradient.contents.data[i], expected[i], atol=1e-6)


# def test_backward_linear(tlib, net_lin_sig):
#     net = net_lin_sig
#
#     InputShape = ctypes.c_int * 2
#     InputData = ctypes.c_float * 5
#     input_shape = InputShape(5, 1)
#     input_data = InputData(1., 2., 3., 4., 5.)
#     input_tensor = tlib.init_tensor(2,input_shape, input_data)
#
#     WeightShape = ctypes.c_int * 2
#     WeightData = ctypes.c_float * 5
#     weight_shape = WeightShape(1, 5)
#     weight_data = WeightData(0.1691, 0.0883, -0.3497, 0.3626, -0.0421)
#     weight_tensor = tlib.init_tensor(2,weight_shape, weight_data)
#
#     BiasShape = ctypes.c_int * 2
#     BiasData = ctypes.c_float * 1
#     bias_shape = BiasShape(1, 1)
#     bias_data = BiasData(-0.3685)
#     bias_tensor = tlib.init_tensor(2,bias_shape, bias_data)
#
#     # TODO
#     # backprop for subtraction layer not yet implemented
