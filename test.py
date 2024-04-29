from ctypes import CDLL
import ctypes
import numpy as np

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


def conv2d_3x3mean_kernel():
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
    kernel_shape = ImageShape(1,1,3,3)
    kernel_data = ImageData(
        1/9,1/9,1/9,
        1/9,1/9,1/9,
        1/9,1/9,1/9
    )
    kernel = tlib.init_tensor(ctypes.c_int(4),kernel_shape, kernel_data)
    out = tlib.op_fp32conv2d(image, kernel, None, 1, 0)

    expected = [3,4,5,4,5,6,5,6,7]

    for i in range(9):
        if not np.allclose(out.contents.data[i], expected[i]):
            raise Exception("Test failed. Out at {i} is out.contents.data[i], while it was expected to be expected[i]")
    print("Success!")

conv2d_3x3mean_kernel()
