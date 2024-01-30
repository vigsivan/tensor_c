#pragma once
#define getindex(t,...) op_fp32getindex(t, t->ndims, __VA_ARGS__)
#define setindex(t, v, ...) op_fp32setindex(t, v, t->ndims, __VA_ARGS__)

typedef struct tensor_fp32{
	int size;
	int ndims;
    int* dims;
    int* strides;
    float* data;
} tensor_fp32;

/*
 * Constructor and destructor
 */
tensor_fp32* init_with_data(int ndims, int* dims, float* data);
tensor_fp32* init_with_zeros(int ndims, int* dims);
tensor_fp32* init_with_random(int ndims, int* dims);
tensor_fp32* init_tensor(int ndims, int* dims);
void free_tensor(tensor_fp32* t);

/*
 * Get and Set Index
 */
float op_fp32getindex(tensor_fp32* t, int ndims, ...);
void op_fp32setindex(tensor_fp32* t, float val, int ndims, ...);

/*
 * Scalar Operations
 */
tensor_fp32* scalarop_fp32mul(tensor_fp32* t, float scalar);
tensor_fp32* scalarop_fp32pad2d(tensor_fp32* t, int padh, int padw, float padval);

/*
 * Tensor Operations
 */
tensor_fp32* op_fp32add(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32sub(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32dot(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32linear(tensor_fp32* t, tensor_fp32* w, tensor_fp32* b);

/*
 * Window Operations
 */
tensor_fp32* op_fp32conv2d(tensor_fp32* t, tensor_fp32* k, int stride, int padding);
tensor_fp32* op_fp32maxpool2d(tensor_fp32* t, int kh, int kw, int stride, int padding);
tensor_fp32* op_fp32avgpool2d(tensor_fp32* t, int kh, int kw, int stride, int padding);

/*
 * Shape Operations
 */
tensor_fp32* op_fp32flatten(tensor_fp32* t);
tensor_fp32* op_fp32transposelinear(tensor_fp32* t);

/*
 * Activation Functions
 */
tensor_fp32* op_fp32relu(tensor_fp32* t);
tensor_fp32* op_fp32sigmoid(tensor_fp32* t);


/*
 * Scalar Inplace Operations
 */
void scalarop_inplace_fp32mul(tensor_fp32* t, float scalar);
void scalarop_inplace_fp32add(tensor_fp32* t, float scalar);

/*
 * Debugging
 */
void print_2d(tensor_fp32* t);
void print_linear(tensor_fp32* t);
