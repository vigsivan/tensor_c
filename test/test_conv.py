import pytest

@pytest.fixture()
def fivebyfivearr(create_ctensor):
    return create_ctensor([
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9], 1,1,5,5)

def test_conv2d_3x3mean_kernel(tlib, check, fivebyfivearr, create_ctensor):
    kernel = create_ctensor([1/9]*9, 1,1,3,3)
    out = tlib.op_fp32conv2d(fivebyfivearr, kernel, None, 1, 0)
    expected = [3,4,5,4,5,6,5,6,7]
    check(out,expected)

def test_conv2d_2x2mean_kernel(tlib, check, fivebyfivearr, create_ctensor):
    kernel = create_ctensor([1/4]*4, 1,1,2,2)
    out = tlib.op_fp32conv2d(fivebyfivearr, kernel, None, 1, 0)
    expected = [
        2,3,4,5,
        3,4,5,6,
        4,5,6,7,
        5,6,7,8
    ]
    check(out, expected)

def test_multi_channel_conv(tlib, check, fivebyfivearr, create_ctensor):
    kernel = create_ctensor([*[1/9]*9, *[1/18]*9], 2, 1, 3, 3)
    out = tlib.op_fp32conv2d(fivebyfivearr, kernel, None, 1, 0)
    expected = [3,4,5,4,5,6,5,6,7,1.5,2,2.5,2,2.5,3,2.5,3,3.5]
    check(out,expected)

