# Description: This script is used to serialize a model's state_dict to a file.
# Usage: python serialize_state_dict.py <model_path> <output_path>
# Inspired by (and in some cases directly lifted from) Karpathy's excellent llama2.c code

import os
import struct
import sys
import torch

def main(model_path, output_path):
    model = torch.load(model_path)
    nmodules = len(model.keys())
    tensors = list(model.values())
    dimsum = sum(len(t.shape) for t in tensors)
    ndims = [len(t.shape) for t in tensors]
    dimsconcat = [s for t in tensors for s in t.shape]
    out_file = open(output_path, 'wb')
    header = struct.pack('i'*(1+nmodules+dimsum), nmodules,*ndims,*dimsconcat)
    out_file.write(header)
    for tensor in tensors:
        serialize_fp32(out_file, tensor)
    out_file.close()

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python serialize_state_dict.py <model_path> <output_path>')
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f'Error: {sys.argv[1]} does not exist')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
