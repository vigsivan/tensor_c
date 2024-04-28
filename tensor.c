#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

/**************************************************
 * Constructor and destructor
 **************************************************/
tensor_fp32* init_tensor(int ndims, int* dims, float* data){
    if(dims == NULL){
        printf("Error: dims is NULL\n");
        exit(1);
    }
    tensor_fp32 *t = (tensor_fp32* ) malloc(sizeof(tensor_fp32));
	int size = 1;
	for (int i = 0; i < ndims; i++) {
		size *= dims[i];
	}
    t->size = size;
	t->ndims = ndims;
    t->strides = NULL;
    t->dims = malloc(ndims * sizeof(int));
    for(int i=0; i<ndims; i++){
	t->dims[i] = dims[i];
    }
    t->data = calloc(size, sizeof(float));
    if (data != NULL){
        for(int i=0; i<size; i++){
            t->data[i] = data[i];
        }
    }
    return t;
}

tensor_fp32* init_empty_tensor(int ndims, ...){
    va_list indexes;
    va_start(indexes, ndims);
    int dims[ndims];
    for (int i = 0; i < ndims; i++){
        dims[i] = va_arg(indexes, int);
    }
    return init_tensor(ndims, dims, NULL);
}

void free_tensor(tensor_fp32* t){
    free(t->data);
    free(t->dims);
    free(t->strides);
    free(t);
}

/**************************************************
 * Get and Set Index
 **************************************************/

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
        printf("Get Error: index %d is out of bounds for tensor of size %d\n", idx, t->size);
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
        printf("Set Error: index %d is out of bounds for tensor of size %d\n", idx, t->size);
        exit(1);
    }
    t->data[idx] = val;
}

/**************************************************
 * Scalar Operations
 **************************************************/

tensor_fp32* scalarop_fp32mul(tensor_fp32* t, float scalar){
	tensor_fp32* t2 = init_tensor(t->ndims, t->dims, NULL);
	int size = 1;
	for (int i = 0; i < t->ndims; i++) {
		size *= t->dims[i];
	}
	for (int i=0; i < size; i++) {
		t2->data[i] = t->data[i] * scalar;
	}
    return t2;
}


tensor_fp32* scalarop_fp32pad2d(tensor_fp32* t, int padh, int padw, float padval){
    if(t->ndims != 4){
        printf("Error: scalarop_fp32pad2d expects 4d input tensor");
        exit(1);
    }
    tensor_fp32* padded = T(t->dims[0],t->dims[1],(padh*2)+t->dims[2],(padw*2)+t->dims[3]);
    scalarop_inplace_fp32add(padded, padval);
    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c<t->dims[1]; c++){
            for (int h=0; h<t->dims[2]; h++){
                for (int w=0; w<t->dims[3]; w++){
                    // setindex(padded,getindex(t, n, c, h, w), n, c, h+padh, w+padh) = ;
                    padded->data[
                        (n * padded->dims[1] * padded->dims[2] * padded->dims[3]) +
                        (c * padded->dims[2] * padded->dims[3]) +
                        ((h+padh) * padded->dims[3]) + 
                        w+padh
                    ] = getindex(t, n, c, h, w);
                }
            }
        }
    }
    printf("Padding complete.\n");
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

    tensor_fp32* out = T(t->dims[0], t->dims[1], ho, wo);

    if (padding > 0){
        t = scalarop_fp32pad2d(t, padding, padding, -INFINITY);
    }

    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c < t-> dims[1]; c++) {
            for (int h=laddh; h < t->dims[2] - raddh; h+=stride){
                for (int w=laddw; w < t->dims[3] - raddw; w+=stride){
                    float agg = 0.;
                    int count = 0;
                    for (int kh=h-laddh; kh <= h + raddh; kh++){
                        for (int kw=w-laddw; kw <= w + raddw; kw++){
                            agg += getindex(t,n,c,kh,kw);
                            count += 1;
                        }
                    }

                    float avg = agg / count;
                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    setindex(out, avg, n, c, hindex, windex);
                }
            }
        }
    }
    
    if (padding > 0){
        free_tensor(t);
    } 
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
                            if (getindex(t,n,c,kh,kw) > max_val){
                                max_val = getindex(t,n,c,kh,kw);
                            }
                        }
                    }

                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    setindex(out, max_val, n, c, hindex, windex);
                }
            }
        }
    }
    
    if (padding > 0){
        free_tensor(t);
    } 
    return out;
}

/*
 * Implements 2D convolution (note: actually, correlation) naively.
 * This function is only implemented for a single filter.
 * output tensor has shape (N, Cin, ho, wo) (output shape depends on padding)
 * this function is probably really inefficient.
 * t: input tensor with shape (N, Cin, H, W)
 * k: kernel tensor with 2d shape (Cout, Cin, h, w)
 * b: kernel bias tensor with 2d shape (Cout, 1)
 * stride: stride of convolution
 * padding: padding of convolution. Note: only 0-padding is supported
 */
tensor_fp32* op_fp32conv2d(tensor_fp32* t, tensor_fp32* k, tensor_fp32* b, int stride, int padding){
    if(t->ndims != 4){
        fprintf(stderr, "Error: op_fp32conv2d expects 4d input tensor");
        exit(EXIT_FAILURE);
    }
    if(k->ndims != 4){
        fprintf(stderr, "Error: op_fp32conv2d expects kernel with 4 dims (c,h,w). Got %d", k->ndims);
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

    for (int n = 0; n < batch; n++){
        for (int h = laddh; h < t->dims[2]-raddh; h+=stride){
            for (int w = laddw; w < t->dims[3]-raddw; w+=stride){
                for (int cout=0; cout < out_channels; cout++){
                    float agg = 0;
                    for (int cin=0; cin < in_channels; cin++){
                        for(int hk=0; hk<k->dims[2]; hk++){
                            for(int wk=0; wk<k->dims[3]; wk++){
                                agg += getindex(k, cout, cin, hk, wk) * 
                                    getindex(t, n, cin, h+hk-laddh, w+wk-laddw);
                            }
                        }
                    }

                    // TODO: test the bias term
                    if (b != NULL){
                        agg += getindex(b, cout);
                    }
                    int hindex = floor((h-laddh)/stride);
                    int windex = floor((w-laddw)/stride);
                    setindex(out, agg, n, cout, hindex, windex);

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
    int new_shape[2] = {t->dims[0], 1};
    for (int i=1; i < t->ndims; i++){
        new_shape[1] *= t->dims[i];
    }
    tensor_fp32* out = init_tensor(2, new_shape, NULL);
    out->data = t->data;
    return out;
}

tensor_fp32* op_fp32transposelinear(tensor_fp32* t){
    if(t->ndims != 2){
        printf("Error: op_fp32transposelinear expects 2d input tensor");
        exit(1);
    }
    int new_shape[2] = {t->dims[1], t->dims[0]};
    tensor_fp32* out = init_tensor(2, new_shape, NULL);
    for (int n=0; n < t->dims[0]; n++){
        for (int c=0; c < t->dims[1]; c++){
            setindex(out, getindex(t, n, c), c, n);
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
    for (int i=0; i < t->size; i++){
        out->data[i] = 1 / (1 + (float) exp(-1 * i));
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
    t->data[i] = l->data[i] + r->data[i];
    }
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
        printf("Error: op_fp32linear expects input and weight dims to match");
        exit(1);
    }
    if(b != NULL && w_t->dims[1] != b->dims[0]){
        printf("Error: op_fp32linear expects weight and bias dims to match");
        exit(1);
    }
    int new_shape[2] = {t->dims[0], w_t->dims[1]};
    tensor_fp32* out = init_tensor(2, new_shape, NULL);
    for (int n=0; n < t->dims[0]; n++){
        for (int c=0; c < w_t->dims[1]; c++){
            float res = 0;
            for (int i=0; i < t->dims[1]; i++){
                res += getindex(t, n, i) * getindex(w_t, i, c);
            }
            if (b != NULL){
                res += b->data[c];
            }
            setindex(out, res, n, c);
        }
    }
    free_tensor(w_t);
    return out;
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
                    printf("%f\t", getindex(t, n, c, h, w));
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
        printf("%iD tensor ", t->ndims);
        printf("(");
        for (int d=0; d<t->ndims; d++){
            if (d == t->ndims-1){
                printf("%i): ", t->dims[d]);
            } else { printf("%ix", t->dims[d]); }
        }
        print_raw(t);
        return;
    }
    for (int n=0; n < t->dims[0]; n++){
        if (n == 0){ printf("⎡ "); }
        else if (n==t->dims[0]-1){ printf("⎣ "); }
        else{ printf("| "); }
        for (int c=0; c < t->dims[1]; c++){
            printf("%f ", getindex(t, n, c));
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
