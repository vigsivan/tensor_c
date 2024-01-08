#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"

void print_data(tensor_fp32 *t);
// void print_tensor2d(tensor2d *t);

int main(){
    srand(42);
    int kernel_shape[3] = {3, 3,3};
    int image_shape[4] = {1, 1, 5, 5};

    tensor_fp32 *a = init_with_random(4, image_shape);

    // create a 3x3 mean kernel
    // tensor_fp32 *k = init_with_zeros(3, kernel_shape);
    // scalarop_inplace_fp32add(k, 1);
    // scalarop_inplace_fp32mul(k, (float) 1/9);
    //
    float data[] = {
        1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,
    };
    tensor_fp32* k = init_with_data(3, kernel_shape, data) ;
    // tensor_fp32 *k = init_with_zeros(3, kernel_shape);
    // scalarop_inplace_fp32add(k, 1);
    // scalarop_inplace_fp32mul(k, (float) 1/9);


    printf("Input array:\n");
    print_2d(a);

    printf("3x3 Kernel array:\n");
    print_data(k);

    tensor_fp32* out = op_fp32conv2d(a, k, 1, 1);

    printf("Output array:\n");
    print_2d(out);
    free(a); free(k); free(out);
    return 0;
}

void print_data(tensor_fp32 *t){
    int size = 1;
    for(int i=0; i<t->ndims; i++){
    size *= t->dims[i];
    }
    for(int i=0; i<size; i++){
    printf("%f ", t->data[i]);
    }
    printf("\n");
}
