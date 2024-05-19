import ctypes
import numpy as np

import torch

def test_sigmoid_activation(tlib):
    ImageShape = ctypes.c_size_t * 4
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
    ImageShape = ctypes.c_size_t * 4
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

    KernelShape = ctypes.c_size_t * 4
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
    ImageShape = ctypes.c_size_t * 4
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

    KernelShape = ctypes.c_size_t * 4
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
    ImageShape = ctypes.c_size_t * 4
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

    KernelShape = ctypes.c_size_t * 4
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

