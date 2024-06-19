#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "string.h"
#include "tensor.h"

/**************************************************
 * Constructor and destructor
 **************************************************/
tensor_fp32* init_tensor(size_t ndims, size_t* dims, float* data){
    if(dims == NULL){
        printf("Error: dims is NULL\n");
        exit(1);
    }
    tensor_fp32 *t = (tensor_fp32* ) malloc(sizeof(tensor_fp32));
	size_t size = 1;
	for (int i = 0; i < ndims; i++) {
		size *= dims[i];
	}
    t->size = size;
	t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(size_t));
    memcpy(t->dims, dims, sizeof(size_t) * ndims);
    t->data = calloc(size, sizeof(float));
    if (data != NULL){
        memcpy(t->data, data, size * sizeof(float));
    }
    t->gradient = NULL;
    Op op=Op_none;
    t->children = NULL;
    t->requires_grad=false;
    
    return t;
}

tensor_fp32* init_empty_tensor(size_t ndims, ...){
    va_list indexes;
    va_start(indexes, ndims);
    size_t dims[ndims];
    for (int i = 0; i < ndims; i++){
        dims[i] = va_arg(indexes, size_t);
    }
    return init_tensor(ndims, dims, NULL);
}

void free_tensor(tensor_fp32* t){
    free(t->data);
    free(t->dims);
    free(t->children);
    free(t->gradient);
    free(t);
}

void backward(tensor_fp32* t){
    if (!t->requires_grad){
        fprintf(stderr, "Backward called for tensor that does not require gradient\n");
        exit(1);
    }
    if (t->op == Op_none) {
        fprintf(stderr, "Tensor has op_none, cannot compute backward\n");
        exit(1);
    }
    if (t->size > 1){
        fprintf(stderr, "Can only call backward on a one-item tensor\n");
        exit(1);
    }
    t->gradient = SCALAR(1);
    recursive_backprop(t);
}

void recursive_backprop(tensor_fp32* t){
    if (!t->requires_grad){
        return;
    }
    if (t->gradient == NULL){
        fprintf(stderr, "Expect recursive gradient function to be called with gradient");
    }
    if (t->children != NULL){
        switch (t->op){
            case Op_fp32linear:
                {
                    tensor_fp32* x = t->children[0];
                    tensor_fp32* w = t->children[1];
                    tensor_fp32* b = t->children[2];
                    backwardop_fp32linear(t, x, w, b);
                    recursive_backprop(w);
                    recursive_backprop(b);
                    recursive_backprop(x);
                    break;
                }
            case Op_fp32total:
                {
                    backwardop_fp32total(t);
                    recursive_backprop(t->children[0]);
                    break;
                } 
            case Op_scalarfp32exp:
                {
                    backwardop_scalarfp32exp(t);
                    recursive_backprop(t->children[0]);
                    break; 
                }

            case Op_fp32add:
                {
                    backwardop_fp32add(t);
                    recursive_backprop(t->children[0]);
                    recursive_backprop(t->children[1]);
                    break;
                }
            case Op_fp32sub:
                {
                    backwardop_fp32sub(t);
                    recursive_backprop(t->children[0]);
                    recursive_backprop(t->children[1]);
                    break;
                }
            case Op_fp32sigmoid:
                {
                    backwardop_fp32sigmoid(t);
                    recursive_backprop(t->children[0]);
                    break;
                }
            case Op_fp32conv2d:
                {
                    backwardop_fp32conv2d(t);
                    recursive_backprop(t->children[1]);
                    recursive_backprop(t->children[2]);
                    recursive_backprop(t->children[0]);
                    break;
                }

            case Op_fp32flatten:
                {
                    backwardop_fp32flatten(t);
                    recursive_backprop(t->children[0]);
                    break;
                }
            case Op_fp32pad2d:
                {
                    backwardop_fp32pad2d(t);
                    recursive_backprop(t->children[0]);
                    break;
                }
            case Op_fp32avgpool2d:
                {
                    backwardop_fp32avgpool2d(t);
                    recursive_backprop(t->children[0]);
                    break;
                }
            case Op_fp32mul:
            case Op_fp32dot:
            case Op_fp32maxpool2d:
            case Op_fp32relu:
            case Op_scalarfp32mul:
            case Op_scalarfp32pad2d:
                {
                    fprintf(stderr, "Backprop not yet implemented\n");
                    exit(1);
                }
            case Op_none:
                return;
        }
    }
}


float op_fp32getindex(tensor_fp32* t, int ndims, ...){
    va_list indexes;
    va_start(indexes, ndims);
    int idx = 0;
    for (int i =0; i < t->ndims; i++){
        int index = va_arg(indexes, int);
        for (int j = i+1; j<t->ndims; j++){
            index *= t->dims[j];
        }
        idx += index;
    }
    if (idx >= t->size) {
        printf("Get Error: index %d is out of bounds for tensor of size %zu\n", idx, t->size);
        exit(1);
    }
    return t->data[idx];
}

void op_fp32setindex(tensor_fp32* t, float val, int ndims, ...){
    va_list indexes;
    va_start(indexes, ndims);
    int idx = 0;
    int record[ndims];
    for (int i =0; i < t->ndims; i++){
        int index = va_arg(indexes, int);
        record[i] = index;
        for (int j = i+1; j<t->ndims; j++){
            index *= t->dims[j];
        }
        idx += index;
    }
    if (idx >= t->size) {
        printf("Set Error: index %d is out of bounds for tensor of size %zu\n", idx, t->size);
        exit(1);
    }
    t->data[idx] = val;
}

void register_op(tensor_fp32* t, Op op, int nchildren, ...){
    if (t->children != NULL){
        fprintf(stderr, "This tensor has already been registered");
        exit(1);
    }
    va_list children;
    va_start(children, nchildren);
    t->children = malloc(sizeof(tensor_fp32*) * nchildren);
    t->op = op;
    for (int i =0; i < nchildren; i++){
        t->children[i] = va_arg(children, tensor_fp32*);
        // Some children can be null (e.g. bias term)
        if (t->children[i]){
            t->requires_grad |= t->children[i]->requires_grad;
        }
    }
}

/**************************************************
 * Scalar Operations
 **************************************************/

tensor_fp32* scalarop_fp32mul(tensor_fp32* t, float scalar){
	tensor_fp32* t2 = init_tensor(t->ndims, t->dims, NULL);
	for (int i=0; i < t2->size; i++) {
		t2->data[i] = t->data[i] * scalar;
	}
    tensor_fp32* S = T(1);
    S->data[0] = scalar;
    REGISTER(t2, Op_scalarfp32mul, t, S);
    return t2;
}

tensor_fp32* scalarop_fp32exp(tensor_fp32* t, float scalar){
	tensor_fp32* t2 = init_tensor(t->ndims, t->dims, NULL);
	for (int i=0; i < t2->size; i++) {
		t2->data[i] = powf(t->data[i], scalar);
	}
    tensor_fp32* S = T(1);
    S->data[0] = scalar;
    REGISTER(t2, Op_scalarfp32exp, t, S);
    return t2;
}


tensor_fp32* scalarop_fp32pad2d(tensor_fp32* t, int padh, int padw, float padval){
    if(t->ndims != 4){
        printf("Error: scalarop_fp32pad2d expects 4d input tensor");
        exit(1);
    }
    tensor_fp32* padded = T(t->dims[0],t->dims[1],(padh*2)+t->dims[2],(padw*2)+t->dims[3]);
    tensor_fp32* pad_tensor = T(3);

    pad_tensor->data[0] = padh;
    pad_tensor->data[1] = padw;
    pad_tensor->data[2] = padval;

    REGISTER(padded, Op_fp32pad2d, t, pad_tensor);
    scalarop_inplace_fp32add(padded, padval);
    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c<t->dims[1]; c++){
            for (int h=0; h<t->dims[2]; h++){
                for (int w=0; w<t->dims[3]; w++){
                    // SET(padded,GET(t, n, c, h, w), n, c, h+padh, w+padh) = ;
                    padded->data[
                        (n * padded->dims[1] * padded->dims[2] * padded->dims[3]) +
                        (c * padded->dims[2] * padded->dims[3]) +
                        ((h+padh) * padded->dims[3]) + 
                        w+padh
                    ] = GET(t, n, c, h, w);
                }
            }
        }
    }
    return padded;
}

/**************************************************
 * Window Operations
 **************************************************/

/** Implements 2D avgpool
 * output tensor has shape (N, 1, ho, wo) (output shape depends on padding)
 * if padding is non-zero, each side is padded with negative infinity for padding.
 * Note: we don't check if the padding is valid (i.e. if you supply a large value
 * for padding, then the output might contain negative infinity values).
 * t: input tensor with shape (N, Cin, H, W)
 * k: kernel tensor with 2d shape (h, w)
 * stride: stride of avgpool
 * padding: padding of avgpool
 */
tensor_fp32* op_fp32avgpool2d(tensor_fp32* t, int kh, int kw, int stride, int padding){
    if(t->ndims != 4){
        printf("Error: op_fp32maxpool2d expects 4d input tensor");
        exit(1);
    }
    if (padding < 0){
        printf("Error: expecting padding to be gte 0. Got %d", padding);
        exit(1);
    }


    int ho = floor((t->dims[2] + 2*padding - (kh-1)-1)/stride + 1);
    int wo = floor((t->dims[3] + 2*padding - (kw-1)-1)/stride + 1);

    int mid_h = floor(kh / 2);
    int mid_w = floor(kw / 2);

    /* Get the mid point of the kernel
     * if kernel size is even, then we decide the midpoint
     * to be right of the center. ladddh and raddh are the
     * number of elements to the left and right of the center
     * we will use to compute the maxpool in the h dimension.
     * Similarly for the w dimension.
     */

    int laddh, raddh, laddw, raddw;
    if (kh % 2 == 0) {
        laddh = mid_h, raddh = mid_h - 1;
    }
    else {
        laddh = mid_h, raddh = mid_h;
    }
    if (kw % 2 == 0) {
        laddw = mid_w, raddw = mid_w - 1;
    }
    else {
        laddw = mid_w, raddw = mid_w;
    }

    if (padding > 0){
        t = scalarop_fp32pad2d(t, padding, padding, -INFINITY);
    }

    tensor_fp32* out = T(t->dims[0], t->dims[1], ho, wo);
    REGISTER(out, Op_fp32avgpool2d, t, SCALAR(kh), SCALAR(kw), SCALAR(stride));

    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c < t-> dims[1]; c++) {
            for (int h=laddh; h < t->dims[2] - raddh; h+=stride){
                for (int w=laddw; w < t->dims[3] - raddw; w+=stride){
                    float agg = 0.;
                    int count = 0;
                    for (int kh=h-laddh; kh <= h + raddh; kh++){
                        for (int kw=w-laddw; kw <= w + raddw; kw++){
                            agg += GET(t,n,c,kh,kw);
                            count += 1;
                        }
                    }

                    float avg = agg / count;
                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    SET(out, avg, n, c, hindex, windex);
                }
            }
        }
    }
    
    // if (padding > 0){
    //     free_tensor(t);
    // } 
    return out;
}


/** Implements 2D maxpool
 * output tensor has shape (N, 1, ho, wo) (output shape depends on padding)
 * if padding is non-zero, each side is padded with negative infinity for padding.
 * Note: we don't check if the padding is valid (i.e. if you supply a large value
 * for padding, then the output might contain negative infinity values).
 * t: input tensor with shape (N, Cin, H, W)
 * k: kernel tensor with 2d shape (h, w)
 * stride: stride of convolution
 * padding: padding of convolution
 */
tensor_fp32* op_fp32maxpool2d(tensor_fp32* t, int kh, int kw, int stride, int padding){
    if(t->ndims != 4){
        printf("Error: op_fp32maxpool2d expects 4d input tensor");
        exit(1);
    }
    if (padding < 0){
        printf("Error: expecting padding to be gte 0. Got %d", padding);
        exit(1);
    }


    int ho = floor((t->dims[2] + 2*padding - (kh-1)-1)/stride + 1);
    int wo = floor((t->dims[3] + 2*padding - (kw-1)-1)/stride + 1);

    int mid_h = floor(kh / 2);
    int mid_w = floor(kw / 2);

    /* Get the mid point of the kernel
     * if kernel size is even, then we decide the midpoint
     * to be right of the center. ladddh and raddh are the
     * number of elements to the left and right of the center
     * we will use to compute the maxpool in the h dimension.
     * Similarly for the w dimension.
     */

    int laddh, raddh, laddw, raddw;
    if (kh % 2 == 0) {
        laddh = mid_h, raddh = mid_h - 1;
    }
    else {
        laddh = mid_h, raddh = mid_h;
    }
    if (kw % 2 == 0) {
        laddw = mid_w, raddw = mid_w - 1;
    }
    else {
        laddw = mid_w, raddw = mid_w;
    }

    tensor_fp32* out = T(t->dims[0], t->dims[1], ho, wo);

    if (padding > 0){
        t = scalarop_fp32pad2d(t, padding, padding, -INFINITY);
    }

    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c < t-> dims[1]; c++) {
            for (int h=laddh; h < t->dims[2] - raddh; h+=stride){
                for (int w=laddw; w < t->dims[3] - raddw; w+=stride){
                    float max_val = -INFINITY;
                    for (int kh=h-laddh; kh <= h + raddh; kh++){
                        for (int kw=w-laddw; kw <= w + raddw; kw++){
                            if (GET(t,n,c,kh,kw) > max_val){
                                max_val = GET(t,n,c,kh,kw);
                            }
                        }
                    }

                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    SET(out, max_val, n, c, hindex, windex);
                }
            }
        }
    }
    
    // if (padding > 0){
    //     free_tensor(t);
    // } 
    return out;
}

/*
 * Implements 2D convolution (note: actually, correlation) naively.
 * This function is only implemented for a single filter.
 * output tensor has shape (N, Cout, ho, wo) (output shape depends on padding)
 * this function is probably really inefficient.
 * t: input tensor with shape (N, Cin, H, W)
 * k: kernel tensor with 2d shape (Cout, Cin, h, w)
 * b: kernel bias tensor with 1d shape (Cout)
 * stride: stride of convolution
 * padding: padding of convolution
 */
tensor_fp32* op_fp32conv2d(tensor_fp32* t, tensor_fp32* k, tensor_fp32* b, int stride, int padding){
    if(t->ndims != 4){
        fprintf(stderr, "Error: op_fp32conv2d expects 4d input tensor");
        exit(EXIT_FAILURE);
    }
    if(k->ndims != 4){
        fprintf(stderr, "Error: op_fp32conv2d expects kernel with 4 dims (c,h,w). Got %zu", k->ndims);
        exit(EXIT_FAILURE);
    }
    if (k->dims[1] != t->dims[1]){
        fprintf(stderr, "Error: op_fp32conv2d kernel second dimension to match number of channels in input tensor");
        exit(EXIT_FAILURE);
    }
    if (padding < 0){
        fprintf(stderr, "Error: expecting padding to be gte 0. Got %d", padding);
        exit(EXIT_FAILURE);
    }
    if (stride < 0){
        fprintf(stderr, "Error: expecting stride to be gte 0. Got %d", stride);
        exit(EXIT_FAILURE);
    }

    if (b != NULL && (b->ndims != 1 || b->dims[0] != k->dims[0])){
        fprintf(stderr, "Error: Something is wrong with the bias term in conv2d");
        exit(EXIT_FAILURE);

    }

    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    int ho = floor((t->dims[2] + 2 * padding - (k->dims[2]-1)-1)/stride + 1);
    int wo = floor((t->dims[3] + 2 * padding - (k->dims[3]-1)-1)/stride + 1);

    int mid_h = floor(k->dims[2] / 2);
    int mid_w = floor(k->dims[3] / 2);

    int laddh, raddh, laddw, raddw;
    if (k->dims[2] % 2 == 0) {
        laddh = mid_h; raddh = mid_h-1;
        // laddh = mid_h; raddh = mid_h;
    }
    else {
        laddh = mid_h; raddh = mid_h;
    }
    if (k->dims[3] % 2 == 0) {
        laddw = mid_w; raddw = mid_w-1;
        // laddw = mid_w; raddw = mid_w;
    }
    else {
        laddw = mid_w; raddw = mid_w;
    }

    if (padding > 0) {
        t = scalarop_fp32pad2d(t, padding, padding, (float) 0);
    }

    int batch = t->dims[0];
    int out_channels = k->dims[0];
    int in_channels = k->dims[1];
    int kernel_size = k->dims[2] * k->dims[3];
    tensor_fp32* out = T(t->dims[0], out_channels, ho, wo);

    // tensor_fp32* stride_tensor = T(1);
    // stride_tensor->data[0] = stride;

    REGISTER(out, Op_fp32conv2d, t, k, b, SCALAR(stride));

    for (int n = 0; n < batch; n++){
        for (int h = laddh; h < t->dims[2]-raddh; h+=stride){
            for (int w = laddw; w < t->dims[3]-raddw; w+=stride){
                for (int cout=0; cout < out_channels; cout++){
                    float agg = 0;
                    for (int cin=0; cin < in_channels; cin++){
                        for(int hk=0; hk<k->dims[2]; hk++){
                            for(int wk=0; wk<k->dims[3]; wk++){
                                agg += GET(k, cout, cin, hk, wk) * 
                                    GET(t, n, cin, h+hk-laddh, w+wk-laddw);
                            }
                        }
                    }

                    if (b != NULL){
                        agg += GET(b, cout);
                    }
                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    SET(out, agg, n, cout, hindex, windex);

                }
            }
        }
    }


    return out;
}

/*
 * Helper function for backward pass of conv2d
 */
tensor_fp32* bop_fp32conv2d(tensor_fp32* t, tensor_fp32* g, int stride){
    if(t->ndims != 4){
        fprintf(stderr, "Error: bop_fp32conv2d expects 4d input tensor");
        exit(EXIT_FAILURE);
    }
    if(g->ndims != 4){
        fprintf(stderr, "Error: bop_fp32conv2d expects kernel with 4 dims (c,h,w). Got %zu", g->ndims);
        exit(EXIT_FAILURE);
    }
    if (g->dims[0] != t->dims[0]){
        fprintf(stderr, "Error: bop_fp32conv2d kernel first dimension doesn't match");
        exit(EXIT_FAILURE);
    }
    if (stride < 0){
        fprintf(stderr, "Error: expecting stride to be gte 0. Got %d", stride);
        exit(EXIT_FAILURE);
    }
    int padding = 0;

    // TODO: look at this
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    if (stride > 1) {
        fprintf(stderr, "Strided convolution backprop not yet supported\n");
    }
    int ho = floor((t->dims[2] + 2 * padding - (g->dims[2]-1)-1)/stride + 1);
    int wo = floor((t->dims[3] + 2 * padding - (g->dims[3]-1)-1)/stride + 1);

    int mid_h = floor(g->dims[2] / 2);
    int mid_w = floor(g->dims[3] / 2);

    int laddh, raddh, laddw, raddw;
    if (g->dims[2] % 2 == 0) {
        laddh = mid_h; raddh = mid_h-1;
        // laddh = mid_h; raddh = mid_h;
    }
    else {
        laddh = mid_h; raddh = mid_h;
    }
    if (g->dims[3] % 2 == 0) {
        laddw = mid_w; raddw = mid_w-1;
        // laddw = mid_w; raddw = mid_w;
    }
    else {
        laddw = mid_w; raddw = mid_w;
    }

    if (padding > 0) {
        t = scalarop_fp32pad2d(t, padding, padding, (float) 0);
    }

    int batch = t->dims[0];
    int out_channels = g->dims[1];
    int in_channels = t->dims[1];
    int kernel_size = g->dims[2] * g->dims[3];
    tensor_fp32* out = T(out_channels, in_channels, ho, wo);

    for (int cout=0; cout < out_channels; cout++){
        for (int h = laddh; h < t->dims[2]-raddh; h+=stride){
            for (int w = laddw; w < t->dims[3]-raddw; w+=stride){
                for (int cin=0; cin < in_channels; cin++){
                    float agg = 0;
                    for (int n = 0; n < batch; n++){
                        for(int hk=0; hk<g->dims[2]; hk++){
                            for(int wk=0; wk<g->dims[3]; wk++){
                                agg += GET(g, n, cout, hk, wk) * 
                                    GET(t, n, cin, h+hk-laddh, w+wk-laddw);
                            }
                        }
                    }

                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    SET(out, agg, cout, cin, hindex, windex);

                }
            }
        }
    }

    return out;
}

/**************************************************
 * Shape Operations
 **************************************************/

/*
 * Implements flattenining operation
 * input tensor has shape (N, *)
 * output tensor has shape (N, D) where D = prod(*)
 * Note: This function just returns a view of the original data.
 */
tensor_fp32* op_fp32flatten(tensor_fp32* t){
    size_t D =  1;
    for (size_t i=1; i < t->ndims; i++){
        D *= t->dims[i];
    }
    tensor_fp32* out = T(t->dims[0], D);
    REGISTER(out, Op_fp32flatten, t);
    memcpy(out->data, t->data, sizeof(float) * t->size);
    return out;
}

tensor_fp32* op_fp32transposelinear(tensor_fp32* t){
    if(t->ndims != 2){
        printf("Error: op_fp32transposelinear expects 2d input tensor");
        exit(1);
    }
    size_t new_shape[2] = {t->dims[1], t->dims[0]};
    tensor_fp32* out = init_tensor(2, new_shape, NULL);
    for (int n=0; n < t->dims[0]; n++){
        for (int c=0; c < t->dims[1]; c++){
            SET(out, GET(t, n, c), c, n);
        }
    }
    return out;
}

/**************************************************
 * Activation Functions
 **************************************************/

tensor_fp32* op_fp32relu(tensor_fp32* t){
    tensor_fp32* out = init_tensor(t->ndims, t->dims, NULL);
    for (int i=0; i < t->size; i++){
        if (t->data[i] < 0){
            out->data[i] = 0;
        }
        else{
            out->data[i] = t->data[i];
        }
    }
    return out;
}

tensor_fp32* op_fp32sigmoid(tensor_fp32* t){
    tensor_fp32* out = init_tensor(t->ndims, t->dims, NULL);
    REGISTER(out, Op_fp32sigmoid, t);
    for (int i=0; i < t->size; i++){
        out->data[i] = 1 / (1 + exp(-1 * t->data[i]));
    }
    return out;
}

/**************************************************
 * Scalar Inplace Operations
 **************************************************/

void scalarop_inplace_fp32mul(tensor_fp32* t, float scalar){
    for (int i=0; i < t->size; i++) {
        t->data[i] = t->data[i] * scalar;
    }
}

void scalarop_inplace_fp32add(tensor_fp32* t, float scalar){
    for (int i=0; i < t->size; i++) {
        t->data[i] = t->data[i] + scalar;
    }
}

/**************************************************
 * Tensor Operations
 **************************************************/

tensor_fp32* op_fp32add(tensor_fp32* l, tensor_fp32* r){
    if(l->ndims != r->ndims){
        fprintf(stderr, "Error: ndims of l and r are not equal\n");
        exit(1);
    }
    for(int i=0; i<l->ndims; i++){
        if(l->dims[i] != r->dims[i]){
            printf("Error: dims of l and r are not equal\n");
            exit(1);
        }
    }
    tensor_fp32 *t = init_tensor(l->ndims, l->dims, NULL);
    for(int i=0; i<t->size; i++){
        t->data[i] = l->data[i] + r->data[i];
    }
    REGISTER(t, Op_fp32add, l, r);
    return t;
}


tensor_fp32* op_fp32sub(tensor_fp32* l, tensor_fp32* r){
    if(l->ndims != r->ndims){
    printf("Error: ndims of l and r are not equal\n");
    exit(1);
    }
    for(int i=0; i<l->ndims; i++){
    if(l->dims[i] != r->dims[i]){
        printf("Error: dims of l and r are not equal\n");
        exit(1);
    }
    }
    tensor_fp32 *t = init_tensor(l->ndims, l->dims, NULL);
    int size = 1;
    for(int i=0; i<l->ndims; i++){
    size *= l->dims[i];
    }
    for(int i=0; i<size; i++){
    t->data[i] = l->data[i] - r->data[i];
    }
    REGISTER(t, Op_fp32sub, l, r);
    return t;
}

/**
 * Implements linear layer
 * input tensor has shape (N, Cin)
 * weight tensor has shape (Cout, Cin) (Thus the weight tensor is transposed prior to computation)
 * bias tensor has shape (Cout). bias can be NULL
 * output tensor has shape (N, Cout)
 */
tensor_fp32* op_fp32linear(tensor_fp32* t, tensor_fp32* w, tensor_fp32* b){
    if(t->ndims != 2){
        printf("Error: op_fp32linear expects 2d input tensor");
        exit(1);
    }
    if(w->ndims != 2){
        printf("Error: op_fp32linear expects 2d weight tensor");
        exit(1);
    }

    tensor_fp32* w_t = op_fp32transposelinear(w);

    if(b != NULL && b->ndims != 1){
        printf("Error: op_fp32linear expects 1d bias tensor (if provided)");
        exit(1);
    }
    if(t->dims[1] != w_t->dims[0]){
        printf("Error: op_fp32linear expects input and weight dims to match. In dim 1 is %zu and weight dim 0 is %zu", t->dims[1], w_t->dims[0]);
        exit(1);
    }
    if(b != NULL && w_t->dims[1] != b->dims[0]){
        printf("Error: op_fp32linear expects weight and bias dims to match");
        exit(1);
    }
    tensor_fp32* out = T(t->dims[0], w_t->dims[1]);
    REGISTER(out, Op_fp32linear, t, w, b);

    for (int n=0; n < t->dims[0]; n++){
        for (int c=0; c < w_t->dims[1]; c++){
            float res = 0;
            for (int i=0; i < t->dims[1]; i++){
                res += GET(t, n, i) * GET(w_t, i, c);
            }
            if (b != NULL){
                res += b->data[c];
            }
            SET(out, res, n, c);
        }
    }
    free_tensor(w_t);
    return out;
}

tensor_fp32* op_fp32total(tensor_fp32* t){
    tensor_fp32* out = T(1);
    for (int i = 0; i < t-> size; i++){
        out->data[0] += t->data[i];
    }
    REGISTER(out, Op_fp32total, t);
    return out;
}

/**************************************************
 * Backprop
 **************************************************/

void backwardop_fp32conv2d(tensor_fp32* out){
    tensor_fp32* t = out->children[0];
    tensor_fp32* kw = out->children[1];
    tensor_fp32* kb = out->children[2];
    tensor_fp32* str = out->children[3];
    int stride = (int) str->data[0];
    if (kb != NULL){
        kb->gradient = init_tensor(kb->ndims, kb->dims, NULL);
        for (int cout = 0; cout < kb->dims[0]; cout++){
            float agg = 0;
            for (int n = 0; n < out->dims[0]; n++){
                for (int h = 0; h < out->dims[2]; h++){
                    for (int w = 0; w < out->dims[3]; w++){
                        kb->gradient->data[cout] += GET(out->gradient, n, cout, h, w);
                    }
                }
            }

        }
    }
    // TODO: how to incorporate stride?
    kw->gradient = bop_fp32conv2d(t, out->gradient, stride);

    // rotate by 180 degrees (i.e. perform correlation)
    // and switch axes
    tensor_fp32* kernel = T(kw->dims[1], kw->dims[0], kw->dims[2], kw->dims[3]);
    for (int cout=0; cout < kw->dims[0]; cout++){
        for (int cin=0; cin < kw->dims[1]; cin++){
            for (int h=0; h < kw->dims[2]; h++){
                for (int w=0; w < kw->dims[3]; w++){ SET(kernel, 
                            GET(kw, cout, cin, h, w), 
                            cin, cout, kw->dims[2]-h-1, kw->dims[3]-w-1);
                }
            }
        }
    }

    // TODO: check padding
    // FIXME: assuming stride is 1
    int ho = t->dims[2];
    int hin = out->dims[2];
    int kh = kw->dims[2];
    int p = (int) floor(0.5 * (stride*(ho-1) + 1 + (kh-1) - hin));
    t->gradient = op_fp32conv2d(out->gradient, kernel, NULL, 1, p);

}

void backwardop_fp32avgpool2d(tensor_fp32* out){
    tensor_fp32* child = out->children[0];
    int kh = (int) out->children[1]->data[0];
    int kw = (int) out->children[2]->data[0];
    int stride = (int) out->children[3]->data[0];

    int mid_h = floor(kh / 2);
    int mid_w = floor(kw / 2);
    int laddh, raddh, laddw, raddw;
    if (kh % 2 == 0) {
        laddh = mid_h, raddh = mid_h - 1;
    }
    else {
        laddh = mid_h, raddh = mid_h;
    }
    if (kw % 2 == 0) {
        laddw = mid_w, raddw = mid_w - 1;
    }
    else {
        laddw = mid_w, raddw = mid_w;
    }

    child->gradient = init_tensor(child->ndims, child->dims, NULL);
    tensor_fp32* grad = out->gradient;

    for (int n=0; n<child->dims[0]; n++){
        for (int c=0; c < child-> dims[1]; c++) {
            for (int h=laddh; h < child->dims[2] - raddh; h+=stride){
                for (int w=laddw; w < child->dims[3] - raddw; w+=stride){
                    float agg = 0.;
                    int count = 0;
                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    float val = GET(grad, n, c, hindex, windex) / (kh * kw);

                    for (int kh=h-laddh; kh <= h + raddh; kh++){
                        for (int kw=w-laddw; kw <= w + raddw; kw++){
                            SETPLUSEQUALS(child->gradient, val, n, c, kh, kw) ;
                        }
                    }
                }
            }
        }
    }
}

void backwardop_fp32pad2d(tensor_fp32* t){
    tensor_fp32* unpadded = t->children[0];
    tensor_fp32* pad_params = t->children[1];
    int padh = pad_params->data[0];
    int padw = pad_params->data[1];

    unpadded->gradient = init_tensor(unpadded->ndims, unpadded->dims, NULL);
    for (int n = 0; n < unpadded->dims[0]; n++){
        for (int c = 0; c < unpadded->dims[1]; c++){
            for (int h = padh; h < unpadded->dims[2]-padh; h++){
                for (int w = padw; w < unpadded->dims[2]-padw; w++){
                    SET(unpadded->gradient, 
                            GET(t->gradient, n, c, h, w),
                            n, c, h-padh, w-padw);
                }
            }
        }
    }
}

void backwardop_fp32flatten(tensor_fp32* t){
    tensor_fp32* child = t->children[0];
    child->gradient = init_tensor(child->ndims, child->dims, NULL);
    for (int i = 0; i < child->size; i++){
        child->gradient->data[i] = t->gradient->data[i];
    }
}

void backwardop_fp32total(tensor_fp32* t){
    tensor_fp32* child = t->children[0];
    child->gradient = init_tensor(child->ndims, child->dims, NULL);
    for (int i = 0; i < child->size; i++){
        child->gradient->data[i] = t->gradient->data[0];
    }
}

void backwardop_fp32linear(tensor_fp32* out, tensor_fp32* t, tensor_fp32* w, tensor_fp32* b){
    b->gradient = out->gradient;
    w->gradient = op_fp32linear(out->gradient, op_fp32transposelinear(t), NULL);
    t->gradient = op_fp32linear(out->gradient, op_fp32transposelinear(w), NULL);
}

void backwardop_scalarfp32exp(tensor_fp32* t){
    // TODO: this isn't clean
    t->children[0]->gradient = scalarop_fp32mul(t->gradient, t->children[1]->data[0]);
}

void backwardop_fp32add(tensor_fp32* t){
    t->children[0]->gradient = t->gradient;
    t->children[1]->gradient = t->gradient;
}

void backwardop_fp32sub(tensor_fp32* t){
    t->children[0]->gradient = t->gradient;
    t->children[1]->gradient = scalarop_fp32mul(t->gradient, -1);
}

void backwardop_fp32sigmoid(tensor_fp32* t){
    tensor_fp32* child = t->children[0];
    child->gradient = init_tensor(t->ndims, t->dims, NULL);
    for (int i=0; i < t->size; i++){
        child->gradient->data[i] = t->data[i] * (1 - t->data[i]) * t->gradient->data[i];
    }
}


/**************************************************
 * Debugging
 **************************************************/

/*
 * Implements printing for 2d tensors
 * with shape (N, C, H, W)
 * if C > 1, then only the first channel is printed.
 * if N > 1, then only the first batch is printed.
 */
void print_2d(tensor_fp32* t){
    if(t->ndims != 4){
        printf("Error: print_2d only works with tensors of shape (N, C, H, W)");
        exit(1);
    }
    for (int n=0; n < t->dims[0]; n++){
        for (int c=0; c < t->dims[1]; c++){
            printf("Batch %d Channel %d:\n", n, c);
            for(int h=0; h < t->dims[2]; h++){
                if (h == 0){ printf("⎡ "); }
                else if (h==t->dims[2]-1){ printf("⎣ "); }
                else{ printf("| "); }

                for (int w=0; w<t->dims[3]; w++){
                    printf("%f\t", GET(t, n, c, h, w));
                }

                if (h == 0){ printf("⎤\n"); }
                else if (h==t->dims[2]-1){ printf("⎦\n"); }
                else{ printf("|\n"); }
            }
        }
    }
}

void print_linear(tensor_fp32* t){
    if (t->ndims != 2){
        printf("%zuD tensor ", t->ndims);
        printf("(");
        for (int d=0; d<t->ndims; d++){
            if (d == t->ndims-1){
                printf("%zu): ", t->dims[d]);
            } else { printf("%zux", t->dims[d]); }
        }
        print_raw(t);
        return;
    }
    for (int n=0; n < t->dims[0]; n++){
        if (n == 0){ printf("⎡ "); }
        else if (n==t->dims[0]-1){ printf("⎣ "); }
        else{ printf("| "); }
        for (int c=0; c < t->dims[1]; c++){
            printf("%f ", GET(t, n, c));
        }
        if (n == 0){ printf("⎤\n"); }
        else if (n==t->dims[0]-1){ printf("⎦\n"); }
        else{ printf("|\n"); }
    }
}

void print_raw(tensor_fp32* t){
    printf("[ ");
    for (int n=0; n < t->size; n++){
        printf("%f ", t->data[n]);
    }
    printf("]\n");
}


