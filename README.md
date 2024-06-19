# A C Tensor Library For building ConvNets

Example usage with ctypes in Python:

```python

    tlib = CDLL("./bin/tensor.so")
    conv_out = tlib.op_fp32conv2d(input_tensor, cnet['conv1.weight'], cnet['conv1.bias'], 1, 0)
    act_out1 = tlib.op_fp32sigmoid(conv_out)
    pool_out = tlib.op_fp32avgpool2d(act_out1, 2,2,2,0)
    flatten_out = tlib.op_fp32flatten(pool_out)
    linear_out= tlib.op_fp32linear(flatten_out, cnet['linear.weight'], cnet['linear.bias'])
    act_out = tlib.op_fp32sigmoid(linear_out)
    loss_out = tlib.op_fp32sub(act_out, target_tensor)
    check(loss_out, loss)

    tlib.backward(loss_out)

```
