import torch

def test_sigmoid_activation(tlib, check, create_ctensor):
    input_torch = torch.Tensor([
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    ])

    image =create_ctensor(input_torch.numpy().squeeze().tolist(), 1,1,5,5)
    out = tlib.op_fp32sigmoid(image);
    expected = torch.nn.Sigmoid()(input_torch)
    check(out, expected)
