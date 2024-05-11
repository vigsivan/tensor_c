import pytest
import ctypes
from ctypes import CDLL

@pytest.fixture(scope="module")
def tlib():
    tlib = CDLL("./bin/tensor.so")

    class TensorFP32(ctypes.Structure): pass
    TensorFP32._fields_ = [
        ("size", ctypes.c_int),
        ("ndims", ctypes.c_int),
        ("dims", ctypes.POINTER(ctypes.c_int)),
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("gradient", ctypes.POINTER(TensorFP32)),
        ("Op", ctypes.c_int),
        ("children", ctypes.POINTER(ctypes.POINTER(TensorFP32))),
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

