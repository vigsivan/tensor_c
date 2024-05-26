#pragma once
#include <stdlib.h>
#include <stdio.h>

#define GET(t,...) op_fp32getindex(t, t->ndims, __VA_ARGS__)
#define SET(t, v, ...) op_fp32setindex(t, v, t->ndims, __VA_ARGS__)
#define NUMARGS(...)  (sizeof((int[]){__VA_ARGS__})/sizeof(int))
#define T(...) init_empty_tensor(NUMARGS(__VA_ARGS__), __VA_ARGS__)
#define ONES(...) init_ones_tensor(NUMARGS(__VA_ARGS__), __VA_ARGS__)
#define RAND(...) init_random_tensor(NUMARGS(__VA_ARGS__), __VA_ARGS__)
#define REGISTER(t,operation,...)                                                    \
    do {                                                                             \
        int ntensors = (sizeof((tensor_fp32*[]){__VA_ARGS__})/sizeof(tensor_fp32*)); \
        register_op(t, operation, ntensors, __VA_ARGS__);                            \
    } while(0)                                                                       \

typedef enum {
    Op_none,
    Op_fp32mul,
    Op_fp32pad2d,
    Op_fp32add,
    Op_fp32sub,
    Op_fp32dot,
    Op_fp32linear,
    Op_fp32conv2d,
    Op_fp32maxpool2d,
    Op_fp32avgpool2d,
    Op_fp32relu,
    Op_fp32sigmoid,
    Op_fp32flatten,
    Op_fp32total,
    Op_scalarfp32exp,
    Op_scalarfp32mul,
    Op_scalarfp32pad2d,
} Op;

typedef struct tensor_fp32{
	size_t size;
	size_t ndims;
    size_t* dims;
    float* data;
    Op op;
    struct tensor_fp32* gradient;
    struct tensor_fp32** children;
} tensor_fp32;

/*
 * Constructor and destructor
 */
tensor_fp32* init_tensor(size_t ndims, size_t* dims, float* data);
tensor_fp32* init_empty_tensor(size_t ndims, ...);
tensor_fp32* init_random_tensor(size_t ndims, ...);
tensor_fp32* init_ones_tensor(size_t ndims, ...);

void free_tensor(tensor_fp32* t);

/*
 * Set opcode and children
 */ 

void register_op(tensor_fp32* t, Op op, int nchildren, ...);
void backward(tensor_fp32* t);
void recursive_backprop(tensor_fp32* t);
int get_num_children(Op op);

/*
 * Get and Set Index
 */
float op_fp32getindex(tensor_fp32* t, int ndims, ...);
void op_fp32setindex(tensor_fp32* t, float val, int ndims, ...);

/*
 * Scalar Operations
 */
tensor_fp32* scalarop_fp32exp(tensor_fp32* t, float scalar);
tensor_fp32* scalarop_fp32mul(tensor_fp32* t, float scalar);
tensor_fp32* scalarop_fp32pad2d(tensor_fp32* t, int padh, int padw, float padval);

/*
 * Tensor Operations
 */
tensor_fp32* op_fp32add(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32sub(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32dot(tensor_fp32* l, tensor_fp32* r);
tensor_fp32* op_fp32linear(tensor_fp32* t, tensor_fp32* w, tensor_fp32* b);
tensor_fp32* op_fp32total(tensor_fp32* t);

/*
 * Window Operations
 */
tensor_fp32* op_fp32conv2d(tensor_fp32* t, tensor_fp32* k, tensor_fp32* b, int stride, int padding);
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
 * Backward Pass Functions
 */

void backwardop_fp32total(tensor_fp32* t);
void backwardop_fp32linear(tensor_fp32* out, tensor_fp32* t, tensor_fp32* w, tensor_fp32* b);
void backwardop_scalarfp32exp(tensor_fp32* out);
void backwardop_fp32add(tensor_fp32* out);
void backwardop_fp32sub(tensor_fp32* out);
void backwardop_fp32sigmoid(tensor_fp32* out);
void backwardop_fp32conv2d(tensor_fp32* out);
// helper method for backward op
tensor_fp32*  bop_fp32conv2d(tensor_fp32* t, tensor_fp32* g, int stride);
void backwardop_fp32flatten(tensor_fp32* t);
void backwardop_fp32pad2d(tensor_fp32* t);


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
void print_raw(tensor_fp32* t);
char* get_shape_str(tensor_fp32* t);
