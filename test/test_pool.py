import torch
def test_avgpool2x2(tlib, check, create_ctensor):
    input_torch = torch.Tensor([
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    ])

    image =create_ctensor(input_torch.numpy().squeeze().tolist(), 1,1,5,5)
    out = tlib.op_fp32avgpool2d(image, 2,2,1,0);
    expected = torch.nn.AvgPool2d(kernel_size=(2,2), stride=(1,1))(input_torch.reshape(1,1,5,5))
    check(out, expected)
