#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

tensor_fp32* init_with_data(int ndims, int* dims, float* data){
    if(dims == NULL){
	printf("Error: dims is NULL\n");
	exit(1);
    }
	if (data == NULL) {
		printf("Error: data is NULL\n");
		exit(1);
	}
    tensor_fp32 *t = (tensor_fp32 *)malloc(sizeof(tensor_fp32));
	int size = 1;
	for (int i = 0; i < ndims; i++) {
		size *= dims[i];
	}
    t->size = size;
	t->ndims = ndims;
    t->strides = NULL;
    t->dims = (int *)malloc(ndims * sizeof(int));
    for(int i=0; i<ndims; i++){
	t->dims[i] = dims[i];
    }
    t->data = (float *)malloc(size * sizeof(float));
    for(int i=0; i<size; i++){
	t->data[i] = data[i];
    }
    return t;
}

tensor_fp32* init_with_zeros(int ndims, int* dims){
    if(dims == NULL){
        printf("Error: dims is NULL\n");
        exit(1);
    }
    tensor_fp32 *t = init_nodata(ndims, dims);
	float* data = (float*)malloc(t->size * sizeof(float));
	for (int i = 0; i < t->size; i++) {
		data[i] = 0.0;
	}
	return init_with_data(ndims, dims, data);
}

tensor_fp32* init_with_random(int ndims, int* dims){
    if(dims == NULL){
	printf("Error: dims is NULL\n");
	exit(1);
    }
	int size = 1;
	for (int i = 0; i < ndims; i++) {
		size *= dims[i];
	}
	float* data = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		data[i] = (float)rand() / RAND_MAX;
	}
	return init_with_data(ndims, dims, data);
}

tensor_fp32* init_nodata(int ndims, int* dims){
	if(dims == NULL){
	printf("Error: dims is NULL\n");
	exit(1);
	}
	tensor_fp32 *t = (tensor_fp32 *)malloc(sizeof(tensor_fp32));
	t->ndims = ndims;
	t->dims = (int *)malloc(ndims * sizeof(int));
	for(int i=0; i<ndims; i++){
        t->dims[i] = dims[i];
	}
    int size = 1;
    for(int i=0; i<ndims; i++){
        size *= dims[i];
    }
    t->data = (float *)malloc(size * sizeof(float));
    t->size = size;
	return t;
}

tensor_fp32* scalarop_fp32mul(tensor_fp32* t, float scalar){
	tensor_fp32* t2 = init_nodata(t->ndims, t->dims);
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

    int new_shape[4] =  {t->dims[0],t->dims[1],(padh*2)+t->dims[2],(padw*2)+t->dims[3]};
    tensor_fp32* padded = init_with_zeros(4, new_shape);
    scalarop_inplace_fp32add(padded, padval);

    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c<t->dims[1]; c++){
            for (int h=0; h<t->dims[2]; h++){
                for (int w=0; w<t->dims[3]; w++){
                    padded->data[
                        (n * padded->dims[1] * padded->dims[2] * padded->dims[3]) +
                        (c * padded->dims[2] * padded->dims[3]) +
                        ((h+padh) * padded->dims[3]) + 
                        w+padh
                    ] = t->data[
                        (n * t->dims[1] * t->dims[2] * t->dims[3]) +
                        (c * t->dims[2] * t->dims[3]) +
                        (h * t->dims[3]) + 
                        w
                    ];

                }
            }
        }
    }
    return padded;
}

float op_fp32getindex4d(tensor_fp32* t, int n, int c, int h, int w){
    return t->data[
        (n * t->dims[1] * t->dims[2] * t->dims[3]) + 
        (c * t->dims[2] * t->dims[3]) + 
        (h * t->dims[3]) + 
        w ];
}


void op_fp32setindex4d(tensor_fp32* t, int n, int c, int h, int w, float val){
    t->data[
        (n * t->dims[1] * t->dims[2] * t->dims[3]) + 
        (c * t->dims[2] * t->dims[3]) + 
        (h * t->dims[3]) + 
        w ] = val;
}


/*
 * Implements 2D maxpool
 * output tensor has shape (N, 1, ho, wo) (output shape depends on padding)
 * if padding is non-zero, each side is padded with negative infinity for padding
 * t: input tensor with shape (N, Cin, H, W)
 * k: kernel tensor with 2d shape (h, w)
 * stride: stride of convolution
 * padding: padding of convolution
 */
tensor_fp32* op_fp32maxpool2d(tensor_fp32* t, int kh, int kw, int stride, int padding){
    if(t->ndims != 4){
        printf("Error: op_fp32conv2d expects 4d input tensor");
        exit(1);
    }
    if (padding < 0){
        printf("Error: expecting padding to be gte 0. Got %d", padding);
        exit(1);
    }

    int ho = floor((t->dims[2] + 2*padding - 1*(kh-1)-1)/stride + 1);
    int wo = floor((t->dims[3] + 2*padding - 1*(kw-1)-1)/stride + 1);

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

    int shape[4] = {t->dims[0], t->dims[1], ho, wo};
    tensor_fp32* out = init_nodata(4, shape);

    if (padding > 0){
        t = scalarop_fp32pad2d(t, padding, padding, -INFINITY);
        // printf("Padded tensor:\n");
        // print_2d(t);
    }

    for (int n=0; n<t->dims[0]; n++){
        for (int c=0; c < t-> dims[1]; c++) {
            for (int h=laddh; h <= t->dims[2]-raddh; h+=stride){
                for (int w=laddw; w <= t->dims[3]-raddw; w+=stride){
                    float max_val = -INFINITY;
                    for (int kh=h-laddh; kh <= h + raddw; kh++){
                        for (int kw=w-laddw; kw <= w + raddw; kw++){
                            if (op_fp32getindex4d(t,n,c,kh,kw) > max_val){
                                max_val = op_fp32getindex4d(t,n,c,kh,kw);
                            }
                        }
                    }
                    // TODO: verify if this is the correct formula for
                    // getting the output indices
                    op_fp32setindex4d(out, n, c, floor((h-laddh)/stride),
                            floor((w-laddw)/stride), max_val);
                }
            }
        }
    }
    
    if (padding > 0){
        free(t);
    } 
    return out;
}

/*
 * Implements 2D convolution (note: actually, correlation) naively.
 * This function is only implemented for a single filter.
 * output tensor has shape (N, 1, ho, wo) (output shape depends on padding)
 * this function is probably really inefficient.
 * t: input tensor with shape (N, Cin, H, W)
 * k: kernel tensor with 2d shape (h, w)
 * stride: stride of convolution
 * padding: padding of convolution. Note: only 0-padding is supported
 */
tensor_fp32* op_fp32conv2d(tensor_fp32* t, tensor_fp32* k, int stride, int padding){
    if(t->ndims != 4){
        printf("Error: op_fp32conv2d expects 4d input tensor");
        exit(1);
    }
    if(k->ndims != 3){
        printf("Error: op_fp32conv2d expects kernel with 3dims (c,h,w). Got %d", k->ndims);
        exit(1);
    }
    if (padding < 0){
        printf("Error: expecting padding to be gte 0. Got %d", padding);
        exit(1);
    }
    
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    int ho = floor((t->dims[2] + 2 * padding - 1 * (k->dims[0]-1)-1)/stride + 1);
    int wo = floor((t->dims[2] + 2 * padding - 1 * (k->dims[0]-1)-1)/stride + 1);

    if (padding > 0) {
        t = scalarop_fp32pad2d(t, padding, padding, (float) 0);
        printf("Padded tensor:\n");
        print_2d(t);
    }

    int mid_h = floor(k->dims[0] / 2);
    int mid_w = floor(k->dims[1] / 2);
    int out_channels = k->dims[0];
    int kernel_size = k->dims[1] * k->dims[2];
    int out_shape[4] = {t->dims[0], out_channels, ho, wo};
    tensor_fp32* out = init_nodata(4, out_shape);
    int out_ptr = 0;
    for (int n=0; n < t->dims[0]; n++){
        for (int oc=0; oc < out_channels; oc++){
            for (int h=mid_h; h < t->dims[2]-mid_h; h+=stride){
                for (int w=mid_w; w < t->dims[3]-mid_w; w+=stride){
                    float res = 0;
                    for (int c = 0; c < t->dims[1]; c++){
                        int k_ptr = oc * kernel_size;
                        for (int kh=h-mid_h; kh <= h + mid_h; kh++){
                            for (int kw=w-mid_w; kw <= w + mid_w; kw++){
                                res += k->data[k_ptr] * t->data[
                                    (n * t->dims[1] * t->dims[2] * t->dims[3]) +
                                    (c * t->dims[2] * t->dims[3]) +
                                    (kh * t->dims[3]) + 
                                    kw
                                    ]; 
                            }
                            k_ptr += 1;
                        }
                    }
                    out->data[out_ptr] = res;
                    out_ptr += 1;
                }
            }
        }
    }

    if (padding > 0){
        free(t);
    }

    return out;
}

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
                    printf("%f ", t->data[
                        (n * t->dims[1] * t->dims[2] * t->dims[3]) +
                        (c * t->dims[2] * t->dims[3]) +
                        (h * t->dims[3]) + 
                        w
                    ]);
                }

                if (h == 0){ printf("⎤\n"); }
                else if (h==t->dims[2]-1){ printf("⎦\n"); }
                else{ printf("|\n"); }
            }
        }
    }
}

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
    tensor_fp32 *t = init_nodata(l->ndims, l->dims);
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
    tensor_fp32 *t = init_nodata(l->ndims, l->dims);
    int size = 1;
    for(int i=0; i<l->ndims; i++){
    size *= l->dims[i];
    }
    for(int i=0; i<size; i++){
    t->data[i] = l->data[i] - r->data[i];
    }
    return t;
}

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
