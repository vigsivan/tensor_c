from ctypes import CDLL
import ctypes
import numpy as np

import pytest

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

    yield tlib

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
    image = tlib.init_tensor(ctypes.c_int(4),image_shape, image_data)

    KernelShape = ctypes.c_int * 4
    KernelData = ctypes.c_float * 9
    kernel_shape = KernelShape(1,1,3,3)
    kernel_data = KernelData(
        1/9,1/9,1/9,
        1/9,1/9,1/9,
        1/9,1/9,1/9
    )
    kernel = tlib.init_tensor(ctypes.c_int(4),kernel_shape, kernel_data)
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
    image = tlib.init_tensor(ctypes.c_int(4),image_shape, image_data)

    KernelShape = ctypes.c_int * 4
    KernelData = ctypes.c_float * 4
    kernel_shape = KernelShape(1,1,2,2)
    kernel_data = KernelData(
        1/4,1/4,
        1/4,1/4,
    )
    kernel = tlib.init_tensor(ctypes.c_int(4),kernel_shape, kernel_data)
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
    image = tlib.init_tensor(ctypes.c_int(4),image_shape, image_data)

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
    kernel = tlib.init_tensor(ctypes.c_int(4),kernel_shape, kernel_data)
    out = tlib.op_fp32conv2d(image, kernel, None, 1, 0)

    expected = [3,4,5,4,5,6,5,6,7,1.5,2,2.5,2,2.5,3,2.5,3,3.5]

    assert out.contents.size == 18

    for i in range(18):
        assert np.allclose(out.contents.data[i], expected[i])

