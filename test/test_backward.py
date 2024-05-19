import pytest
import torch
import ctypes
import numpy as np

@pytest.fixture()
def net_conv_lin_sig():
    class BasicNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
            self.linear = torch.nn.Linear(18, 1)
            self.act = torch.nn.Sigmoid()

        def forward(self, x):
            x = torch.flatten(self.conv(x))
            return self.act(self.linear(x))

    net = BasicNetwork()
    return net

@pytest.fixture()
def net_conv_sig_conv_sig_lin_sig():
    class BasicNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0)
            self.linear = torch.nn.Linear(18, 1)
            self.act = torch.nn.Sigmoid()
            self.intermediate_vars = []

        def forward(self, x):
            x1 = self.act(self.conv1(x))
            x1.retain_grad()
            x1.register_hook(self.set_grad(x1))
            x2 = self.act(self.conv2(x1))
            x2 = torch.flatten(x2)
            return self.act(self.linear(x2))

        def set_grad(self, var):
            def hook(grad):
                var.grad = grad
                self.intermediate_vars.append(var)
            return hook

    net = BasicNetwork()
    return net

@pytest.fixture()
def net_lin_sig():
    class BasicNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = torch.nn.Sigmoid()
            self.linear = torch.nn.Linear(5, 1)

        def forward(self, x):
            return self.act(self.linear(x))

    net = BasicNetwork()
    return net

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

    BiasShape = ctypes.c_int*1
    BiasData = ctypes.c_float * 5
    bias_shape = BiasShape(5)
    bias_data = BiasData(6.,7.,8.,9.,10.,)
    bias_tensor = tlib.init_tensor(1,bias_shape, bias_data)

    bout = tlib.op_fp32add(bias_tensor, input_tensor)
    out = tlib.op_fp32total(bout)
    tlib.backward(out)

    gradient = bias_tensor.contents.gradient
    expected = [1]*5
    assert gradient, "Gradient is None"
    assert gradient.contents.size == 5
    for i in range(gradient.contents.size):
        assert np.allclose(gradient.contents.data[i], expected[i], atol=1e-6)


def test_backward_linear(tlib, net_lin_sig):
    net = net_lin_sig

    input_arr = torch.linspace(1, 5, 5)
    output_arr = net(input_arr)
    target = torch.Tensor([1])
    loss = output_arr - target
    loss.backward()

    InputShape = ctypes.c_int * 2
    InputData = ctypes.c_float * 5
    input_shape = InputShape(1, 5)
    input_data = InputData(1., 2., 3., 4., 5.)
    input_tensor = tlib.init_tensor(2,input_shape, input_data)

    WeightShape = ctypes.c_int * 2
    WeightData = ctypes.c_float * 5
    weight_shape = WeightShape(1, 5)
    weight_data = WeightData(*net.linear.weight.detach().numpy().squeeze().tolist())
    weight_tensor = tlib.init_tensor(2,weight_shape, weight_data)

    BiasShape = ctypes.c_int * 1
    BiasData = ctypes.c_float * 1
    bias_shape = BiasShape(1)
    bias_data = BiasData(net.linear.bias.detach().numpy().squeeze().tolist())
    bias_tensor = tlib.init_tensor(1,bias_shape, bias_data)

    TargetShape = ctypes.c_int * 2
    TargetData = ctypes.c_float * 1
    target_shape = TargetShape(1,1)
    target_data = TargetData(1.)
    target_tensor = tlib.init_tensor(2,target_shape, target_data)

    linear_out= tlib.op_fp32linear(input_tensor, weight_tensor, bias_tensor)
    act_out = tlib.op_fp32sigmoid(linear_out)
    loss_out = tlib.op_fp32sub(act_out, target_tensor)
    tlib.backward(loss_out)

    assert (wgrad := weight_tensor.contents.gradient)
    assert (bgrad := bias_tensor.contents.gradient)

    assert wgrad.contents.size == 5
    assert bgrad.contents.size == 1

    for i in range(wgrad.contents.size):
        assert np.allclose(wgrad.contents.data[i], net.linear.weight.grad[0,i].item(), atol=1e-4)

    assert np.allclose(bgrad.contents.data[0], net.linear.bias.grad[0].item(), atol=1e-4)

def test_backward_conv(tlib, net_conv_lin_sig):
    net = net_conv_lin_sig

    input_arr = torch.ones((1,1,5,5))
    output_arr = net(input_arr)
    target = torch.Tensor([1])
    loss = output_arr - target
    loss.backward()

    InputShape = ctypes.c_int * 4
    InputData = ctypes.c_float * 25
    input_shape = InputShape(1, 1, 5, 5)
    input_data = InputData(*input_arr.numpy().reshape(-1,1).squeeze().tolist())
    input_tensor = tlib.init_tensor(4,input_shape, input_data)

    ConvWShape = ctypes.c_int * 4
    ConvWData = ctypes.c_float * 18
    convw_shape = ConvWShape(2, 1, 3, 3)
    convw_data = ConvWData(*net.conv.weight.detach().numpy().reshape(-1,1).squeeze().tolist())
    convw_tensor = tlib.init_tensor(4, convw_shape, convw_data)

    ConvBShape = ctypes.c_int * 1
    ConvBData = ctypes.c_float * 2
    convb_shape = ConvBShape(2)
    convb_data = ConvBData(*net.conv.bias.detach().numpy().squeeze().tolist())
    convb_tensor = tlib.init_tensor(1, convb_shape, convb_data)

    WeightShape = ctypes.c_int * 2
    WeightData = ctypes.c_float * 18
    weight_shape = WeightShape(1, 18)
    weight_data = WeightData(*net.linear.weight.detach().numpy().squeeze().tolist())
    weight_tensor = tlib.init_tensor(2,weight_shape, weight_data)

    BiasShape = ctypes.c_int * 1
    BiasData = ctypes.c_float * 1
    bias_shape = BiasShape(1)
    bias_data = BiasData(net.linear.bias.detach().numpy().squeeze().tolist())
    bias_tensor = tlib.init_tensor(1,bias_shape, bias_data)

    TargetShape = ctypes.c_int * 2
    TargetData = ctypes.c_float * 1
    target_shape = TargetShape(1,1)
    target_data = TargetData(1.)
    target_tensor = tlib.init_tensor(2,target_shape, target_data)
    
    conv_out = tlib.op_fp32conv2d(input_tensor, convw_tensor, convb_tensor, 1, 0)
    flatten_out = tlib.op_fp32flatten(conv_out)
    linear_out= tlib.op_fp32linear(flatten_out, weight_tensor, bias_tensor)
    act_out = tlib.op_fp32sigmoid(linear_out)
    loss_out = tlib.op_fp32sub(act_out, target_tensor)
    tlib.backward(loss_out)

    assert (wgrad := weight_tensor.contents.gradient)
    assert (bgrad := bias_tensor.contents.gradient)

    assert wgrad.contents.size == 18
    assert bgrad.contents.size == 1

    for i in range(wgrad.contents.size):
        assert np.allclose(wgrad.contents.data[i], net.linear.weight.grad[0,i].item(), atol=1e-4)

    assert np.allclose(bgrad.contents.data[0], net.linear.bias.grad[0].item(), atol=1e-4)

    assert (cwgrad := convw_tensor.contents.gradient)
    assert (cbgrad := convb_tensor.contents.gradient)

    assert cwgrad.contents.size == 18
    assert cbgrad.contents.size == 2

    conv_grad = net.conv.weight.grad.reshape(1,-1)
    for i in range(cwgrad.contents.size):
        assert np.allclose(cwgrad.contents.data[i], conv_grad[0,i].item(), atol=1e-4)

    assert np.allclose(cbgrad.contents.data[0], net.conv.bias.grad[0].item(), atol=1e-4)
    assert np.allclose(cbgrad.contents.data[1], net.conv.bias.grad[1].item(), atol=1e-4)

def test_backward_conv2(tlib, net_conv_sig_conv_sig_lin_sig):
    net = net_conv_sig_conv_sig_lin_sig

    input_arr = torch.ones((1,1,5,5))
    output_arr = net(input_arr)
    target = torch.Tensor([1])
    loss = output_arr - target
    loss.backward()

    InputShape = ctypes.c_int * 4
    InputData = ctypes.c_float * 25
    input_shape = InputShape(1, 1, 5, 5)
    input_data = InputData(*input_arr.numpy().reshape(-1,1).squeeze().tolist())
    input_tensor = tlib.init_tensor(4,input_shape, input_data)

    ConvW1Shape = ctypes.c_int * 4
    ConvW1Data = ctypes.c_float * 18
    convw1_shape = ConvW1Shape(2, 1, 3, 3)
    convw1_data = ConvW1Data(*net.conv1.weight.detach().numpy().reshape(-1,1).squeeze().tolist())
    convw1_tensor = tlib.init_tensor(4, convw1_shape, convw1_data)

    ConvB1Shape = ctypes.c_int * 1
    ConvB1Data = ctypes.c_float * 2
    convb1_shape = ConvB1Shape(2)
    convb1_data = ConvB1Data(*net.conv1.bias.detach().numpy().squeeze().tolist())
    convb1_tensor = tlib.init_tensor(1, convb1_shape, convb1_data)

    ConvWShape = ctypes.c_int * 4
    ConvWData = ctypes.c_float * 36
    convw_shape = ConvWShape(2, 2, 3, 3)
    convw_data = ConvWData(*net.conv2.weight.detach().numpy().reshape(-1,1).squeeze().tolist())
    convw_tensor = tlib.init_tensor(4, convw_shape, convw_data)

    ConvBShape = ctypes.c_int * 1
    ConvBData = ctypes.c_float * 2
    convb_shape = ConvBShape(2)
    convb_data = ConvBData(*net.conv2.bias.detach().numpy().squeeze().tolist())
    convb_tensor = tlib.init_tensor(1, convb_shape, convb_data)

    WeightShape = ctypes.c_int * 2
    WeightData = ctypes.c_float * 18
    weight_shape = WeightShape(1, 18)
    weight_data = WeightData(*net.linear.weight.detach().numpy().squeeze().tolist())
    weight_tensor = tlib.init_tensor(2,weight_shape, weight_data)

    BiasShape = ctypes.c_int * 1
    BiasData = ctypes.c_float * 1
    bias_shape = BiasShape(1)
    bias_data = BiasData(net.linear.bias.detach().numpy().squeeze().tolist())
    bias_tensor = tlib.init_tensor(1,bias_shape, bias_data)

    TargetShape = ctypes.c_int * 2
    TargetData = ctypes.c_float * 1
    target_shape = TargetShape(1,1)
    target_data = TargetData(1.)
    target_tensor = tlib.init_tensor(2,target_shape, target_data)
    
    conv1_out   = tlib.op_fp32conv2d(input_tensor, convw1_tensor, convb1_tensor, 1, 1)
    act1_out    = tlib.op_fp32sigmoid(conv1_out)
    conv2_out   = tlib.op_fp32conv2d(act1_out, convw_tensor, convb_tensor, 1, 0)
    act2_out    = tlib.op_fp32sigmoid(conv2_out)
    flatten_out = tlib.op_fp32flatten(act2_out)
    linear_out  = tlib.op_fp32linear(flatten_out, weight_tensor, bias_tensor)
    act_out     = tlib.op_fp32sigmoid(linear_out)
    loss_out    = tlib.op_fp32sub(act_out, target_tensor)
    tlib.backward(loss_out)

    assert np.allclose(loss.item(), loss_out.contents.data[0])

    assert (wgrad := weight_tensor.contents.gradient)
    assert (bgrad := bias_tensor.contents.gradient)

    assert wgrad.contents.size == 18
    assert bgrad.contents.size == 1

    for i in range(wgrad.contents.size):
        assert np.allclose(wgrad.contents.data[i], net.linear.weight.grad[0,i].item(), atol=1e-4)

    assert np.allclose(bgrad.contents.data[0], net.linear.bias.grad[0].item(), atol=1e-4)

    assert (cwgrad := convw_tensor.contents.gradient)
    assert (cbgrad := convb_tensor.contents.gradient)

    assert cwgrad.contents.size == 36
    assert cbgrad.contents.size == 2

    conv_grad = net.conv2.weight.grad.reshape(1,-1)
    for i in range(cwgrad.contents.size):
        assert np.allclose(cwgrad.contents.data[i], conv_grad[0,i].item(), atol=1e-4)

    assert np.allclose(cbgrad.contents.data[0], net.conv2.bias.grad[0].item(), atol=1e-4)

    assert (cwgrad := convw1_tensor.contents.gradient)
    assert (cbgrad := convb1_tensor.contents.gradient)

    assert cwgrad.contents.size == 18
    assert cbgrad.contents.size == 2

    conv_grad = net.conv1.weight.grad.reshape(1,-1)
    for i in range(cwgrad.contents.size):
        assert np.allclose(cwgrad.contents.data[i], conv_grad[0,i].item(), atol=1e-4)

    assert np.allclose(cbgrad.contents.data[0], net.conv1.bias.grad[0].item(), atol=1e-4)
