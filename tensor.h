#pragma once

typedef struct tensor_fp32{
	int size;
	int ndims;
    int* dims;    /* shape of tensor */
    int* strides; /* strides of tensor (bytes to skip to to get to next value in dimension). */
    float* data; /* data pointer */
} tensor_fp32;

tensor_fp32* init_with_data(int ndims, int* dims, float* data);
tensor_fp32* init_with_zeros(int ndims, int* dims);
tensor_fp32* init_with_random(int ndims, int* dims);
tensor_fp32* init_nodata(int ndims, int* dims);

tensor_fp32* scalarop_fp32mul(tensor_fp32* t, float scalar);
tensor_fp32* scalarop_fp32pad2d(tensor_fp32* t, int padh, int padw);
tensor_fp32* op_fp32add(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32sub(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32mul(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32mul(tensor_fp32* l, tensor_fp32* r);

tensor_fp32* op_fp32conv2d(tensor_fp32* t, tensor_fp32* k, int stride, int padding);

void scalarop_inplace_fp32mul(tensor_fp32* t, float scalar);
void scalarop_inplace_fp32add(tensor_fp32* t, float scalar);

/**
 * 2d tensor functions
 */
void print_2d(tensor_fp32* t);

// tensor_fp32* scalarop_fp32div(tensor_fp32* t, float scalar);

// tensor initializaters
// void zeros_tensor2d(tensor2d *t);
// tensor2d* init_tensor2d(int shape[4]);
// tensor2d* random_tensor2d(tensor2d *t);
// tensor2d* random_like(tensor2d *t);
// tensor2d* zeros_like(tensor2d *t);
//
// tensor2d* add(tensor2d l, tensor2d r);
// tensor2d* subtract(tensor2d l, tensor2d r);


