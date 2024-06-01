import pytest
import ctypes
from ctypes import CDLL

@pytest.fixture(scope="module")
def tlib():
    tlib = CDLL("./bin/tensor.so")

    class TensorFP32(ctypes.Structure): pass
    TensorFP32._fields_ = [
        ("size", ctypes.c_size_t),
        ("ndims", ctypes.c_size_t),
        ("dims", ctypes.POINTER(ctypes.c_size_t)),
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("op", ctypes.c_int),
        ("gradient", ctypes.POINTER(TensorFP32)),
        ("children", ctypes.POINTER(ctypes.POINTER(TensorFP32))),
        ("requires_grad", ctypes.c_bool)
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
    tlib.op_fp32add.argtypes = [TPOINTER,TPOINTER]
    tlib.op_fp32add.restype = TPOINTER
    tlib.op_fp32sub.argtypes = [TPOINTER,TPOINTER]
    tlib.op_fp32sub.restype = TPOINTER


    tlib.op_fp32total.argtypes = [TPOINTER]
    tlib.op_fp32total.restype = TPOINTER

    tlib.backward.argtypes = [TPOINTER]



    yield tlib
