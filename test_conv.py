from ctypes import CDLL
import ctypes
import numpy as np

import pytest
import torch

@pytest.fixture
def tlib():
    tlib = CDLL("./bin/tensor.so")

    class TensorFP32(ctypes.Structure):
        _fields_ = [
                ("size", ctypes.c_int),
                ("ndims", ctypes.c_int),
                ("dims", ctypes.POINTER(ctypes.c_int)),
                ("strides", ctypes.POINTER(ctypes.c_int)),
                ("data", ctypes.POINTER(ctypes.c_float))
        ]

    TPOINTER = ctypes.POINTER(TensorFP32)
    tlib.init_tensor.restype = TPOINTER
    tlib.op_fp32conv2d.argtypes = [TPOINTER, TPOINTER, TPOINTER, ctypes.c_int, ctypes.c_int]
    tlib.op_fp32conv2d.restype = TPOINTER
    tlib.op_fp32sigmoid.restype = TPOINTER
    tlib.exp.restype = ctypes.c_double
    tlib.op_fp32avgpool2d.restype = TPOINTER
    tlib.op_fp32flatten.restype = TPOINTER
    tlib.op_fp32linear.argtypes = [TPOINTER, TPOINTER, TPOINTER]
    tlib.op_fp32linear.restype = TPOINTER

    yield tlib

@pytest.fixture
def lenet_torch():
    class LeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList([
                torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            ])
            self.act = torch.nn.Sigmoid()
            self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            self.linear_layers = torch.nn.ModuleList([
                torch.nn.Linear(400, 120),
                torch.nn.Linear(120, 84),
                torch.nn.Linear(84, 10),
            ])
        def forward(self, x):
            for conv in self.convs:
                x = conv(x)
                x = self.act(x)
                x = self.pool(x)
            x = torch.flatten(x)

            for dense in self.linear_layers:
                x = dense(x)
                x = self.act(x)
            return x

    net = LeNet()
    net.load_state_dict(torch.load("./lenet.pt"))
    net.requires_grad_(False)
    yield net


@pytest.fixture
def lenet(tlib):
    netlib = CDLL("./bin/lenet.so")
    
    class LeNetStruct(ctypes.Structure):
        _fields_ = [
            ("c0w", tlib.init_tensor.restype),
            ("c0b", tlib.init_tensor.restype),
            ("c1w", tlib.init_tensor.restype),
            ("c1b", tlib.init_tensor.restype),
            ("l0w", tlib.init_tensor.restype),
            ("l0b", tlib.init_tensor.restype),
            ("l1w", tlib.init_tensor.restype),
            ("l1b", tlib.init_tensor.restype),
            ("l2w", tlib.init_tensor.restype),
            ("l2b", tlib.init_tensor.restype)
        ]

    netlib.load_lenet.restype = ctypes.POINTER(LeNetStruct)
    netlib.lenet_forward.restype = tlib.init_tensor.restype

    yield netlib

@pytest.fixture
def seed_everything():
    torch.random.manual_seed(42)
    np.random.seed(42)
    yield

def test_lenet(tlib, lenet, lenet_torch, seed_everything):
    arr = torch.randint(0,2,(1,1,28,28)).float()
    net = lenet.load_lenet(b"./lenet.bin")
    ImageShape = ctypes.c_int * 4
    ImageData = ctypes.c_float * (28**2)
    image_shape = ImageShape(1,1,28,28)
    li = torch.flatten(arr).numpy().tolist()
    image_data = ImageData(*li)
    image_c = tlib.init_tensor(4,image_shape, image_data)

    out = lenet.lenet_forward(net, image_c)
    # out = tlib.op_fp32conv2d(image_c, net.contents.c0w, net.contents.c0b, 1, 2)
    # out = tlib.op_fp32sigmoid(out);
    # out = tlib.op_fp32avgpool2d(out, 2, 2, 2, 0)
    #
    # out = tlib.op_fp32conv2d(out, net.contents.c1w, net.contents.c1b, 1, 0)
    # out = tlib.op_fp32sigmoid(out);
    # out = tlib.op_fp32avgpool2d(out, 2, 2, 2, 0)
    # out = tlib.op_fp32flatten(out);
    # out = tlib.op_fp32linear(out, net.contents.l0w, net.contents.l0b)
    # out = tlib.op_fp32sigmoid(out);
    # out = tlib.op_fp32linear(out, net.contents.l1w, net.contents.l1b)
    # out = tlib.op_fp32sigmoid(out);
    # out = tlib.op_fp32linear(out, net.contents.l2w, net.contents.l2b)
    # out = tlib.op_fp32sigmoid(out);

    expected = torch.flatten(lenet_torch(arr)).detach().numpy()

    assert out.contents.size == np.prod(expected.shape)

    for i in range(out.contents.size):
        assert np.allclose(out.contents.data[i], expected[i])

def test_sigmoid_activation(tlib):
    ImageShape = ctypes.c_int * 4
    ImageData = ctypes.c_float * 25
    image_shape = ImageShape(1,1,5,5)
    image_data = ImageData(
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    )
    input_torch = torch.Tensor([
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    ])

    image = tlib.init_tensor(4,image_shape, image_data)
    out = tlib.op_fp32sigmoid(image);
    expected = torch.nn.Sigmoid()(input_torch)

    for i in range(out.contents.size):
        assert np.allclose(out.contents.data[i], expected[i].item(), atol=1e-6)

    


def test_conv2d_3x3mean_kernel(tlib):
    ImageShape = ctypes.c_int * 4
    ImageData = ctypes.c_float * 25
    image_shape = ImageShape(1,1,5,5)
    image_data = ImageData(
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    )
    image = tlib.init_tensor(4,image_shape, image_data)

    KernelShape = ctypes.c_int * 4
    KernelData = ctypes.c_float * 9
    kernel_shape = KernelShape(1,1,3,3)
    kernel_data = KernelData(
        1/9,1/9,1/9,
        1/9,1/9,1/9,
        1/9,1/9,1/9
    )
    kernel = tlib.init_tensor(4,kernel_shape, kernel_data)
    out = tlib.op_fp32conv2d(image, kernel, None, 1, 0)

    expected = [3,4,5,4,5,6,5,6,7]

    assert out.contents.size == 9

    for i in range(9):
        assert np.allclose(out.contents.data[i], expected[i])


def test_conv2d_2x2mean_kernel(tlib):
    ImageShape = ctypes.c_int * 4
    ImageData = ctypes.c_float * 25
    image_shape = ImageShape(1,1,5,5)
    image_data = ImageData(
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    )
    image = tlib.init_tensor(4,image_shape, image_data)

    KernelShape = ctypes.c_int * 4
    KernelData = ctypes.c_float * 4
    kernel_shape = KernelShape(1,1,2,2)
    kernel_data = KernelData(
        1/4,1/4,
        1/4,1/4,
    )
    kernel = tlib.init_tensor(4,kernel_shape, kernel_data)
    out = tlib.op_fp32conv2d(image, kernel, None, 1, 0)

    expected = [
        2,3,4,5,
        3,4,5,6,
        4,5,6,7,
        5,6,7,8
    ]

    assert out.contents.size == 16

    for i in range(16):
        assert np.allclose(out.contents.data[i], expected[i])

# TODO
def test_multi_channel_conv(tlib):
    ImageShape = ctypes.c_int * 4
    ImageData = ctypes.c_float * 25
    image_shape = ImageShape(1,1,5,5)
    image_data = ImageData(
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    )
    image = tlib.init_tensor(4,image_shape, image_data)

    KernelShape = ctypes.c_int * 4
    KernelData = ctypes.c_float * 18
    kernel_shape = KernelShape(2,1,3,3)
    kernel_data = KernelData(
        1/9,1/9,1/9,
        1/9,1/9,1/9,
        1/9,1/9,1/9,
        1/18,1/18,1/18,
        1/18,1/18,1/18,
        1/18,1/18,1/18,

    )
    kernel = tlib.init_tensor(4,kernel_shape, kernel_data)
    out = tlib.op_fp32conv2d(image, kernel, None, 1, 0)

    expected = [3,4,5,4,5,6,5,6,7,1.5,2,2.5,2,2.5,3,2.5,3,3.5]

    assert out.contents.size == 18

    for i in range(18):
        assert np.allclose(out.contents.data[i], expected[i])

