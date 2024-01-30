#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "tensor.h"

void print_data(tensor_fp32 *t);
bool check_equals(float* a, float* b, int size);

/*
 * Test functions
 */
bool maxpool2d_stride1_padding0();
bool maxpool2d_stride2_padding0();
bool maxpool2d_stride1_padding1();
bool linear_layer_test();
bool conv2d_3x3mean_kernel();
bool conv2d_2x2mean_kernel();
bool avgpool2d_stride1_padding0();

int main(){
    srand(42);
    printf("Running tests...\n");
    printf("maxpool2d_stride1_padding0: %s\n", maxpool2d_stride1_padding0() ? "PASSED" : "FAILED");
    printf("maxpool2d_stride2_padding0: %s\n", maxpool2d_stride2_padding0() ? "PASSED" : "FAILED");
    printf("maxpool2d_stride1_padding1: %s\n", maxpool2d_stride1_padding1() ? "PASSED" : "FAILED");
    printf("avgpool2d_stride1_padding0: %s\n", avgpool2d_stride1_padding0() ? "PASSED" : "FAILED");
    printf("linear_layer_test: %s\n", linear_layer_test() ? "PASSED" : "FAILED");
    printf("conv2d_3x3mean_kernel: %s\n", conv2d_3x3mean_kernel() ? "PASSED" : "FAILED");
    printf("conv2d_2x2mean_kernel: %s\n", conv2d_2x2mean_kernel() ? "PASSED" : "FAILED");
    printf("Done.\n");
}

/**
 * Linear layer tests
 */

bool linear_layer_test(){
    int input_shape[2] = {1, 4};
    float input_data[4] = {1,2,3,4};
    tensor_fp32 *input = init_with_data(2, input_shape, input_data);

    int weight_shape[2] = {2, 4};
    float weight_data[8] = {1,2,3,4,5,6,7,8};
    tensor_fp32 *weight = init_with_data(2, weight_shape, weight_data);

    int bias_shape[1] = {2};
    float bias_data[2] = {1,2};
    tensor_fp32 *bias = init_with_data(1, bias_shape, bias_data);

    tensor_fp32 *out = op_fp32linear(input, weight, bias);

    float expected[2] = {31, 72};
    bool passed = true;

    if (!check_equals(out->data, expected, 2)){
        passed = false;
    }

    free_tensor(input);
    free_tensor(weight);
    free_tensor(bias);
    free_tensor(out);

    return passed;
}


/*
 * Maxpool2d tests
 */
bool maxpool2d_stride1_padding0(){
    int image_shape[4] = {1, 1, 4, 4};
    float image_data[16] = { 
        1,2,3,4,
        2,3,4,1,
        3,4,2,1,
        4,1,2,3
    };
    float expected_stride1[9] = { 
        3,4,4,
        4,4,4,
        4,4,3
    };

    tensor_fp32 *a = init_with_data(4, image_shape, image_data);
    tensor_fp32* out = op_fp32maxpool2d(a, 2,2, 1, 0);

    bool passed = true;

    if (!check_equals(out->data, expected_stride1, 9)){
        passed = false;
    }


    free_tensor(a);
    free_tensor(out);

    return passed;
}

bool maxpool2d_stride2_padding0(){
    int image_shape[4] = {1, 1, 4, 4};
    float image_data[16] = { 
        1,2,3,4,
        2,3,4,1,
        3,4,2,1,
        4,1,2,3
    };

    float expected_stride2[4] = { 
        3,4,
        4,3
    };

    bool passed = true;

    tensor_fp32 *a = init_with_data(4, image_shape, image_data);
    tensor_fp32* out = op_fp32maxpool2d(a, 2,2, 2, 0);

    if (!check_equals(out->data, expected_stride2, 4)){
        passed = false;
    }
    
    free_tensor(a);
    free_tensor(out);

    return passed;
}

bool maxpool2d_stride1_padding1(){
    int image_shape[4] = {1, 1, 4, 4};
    float image_data[16] = { 
        1,2,3,4,
        2,3,4,1,
        3,4,2,1,
        4,1,2,3
    };

    float expected_stride1[25] = { 
        1,2,3,4,4,
        2,3,4,4,4,
        3,4,4,4,1,
        4,4,4,3,3,
        4,4,2,3,3
    };

    bool passed = true;

    tensor_fp32 *a = init_with_data(4, image_shape, image_data);
    tensor_fp32* out = op_fp32maxpool2d(a, 2,2, 1, 1);

    if (!check_equals(out->data, expected_stride1, 16)){
        passed = false;
    }
    
    free_tensor(a);
    free_tensor(out);

    return passed;
}

bool avgpool2d_stride1_padding0(){
    int image_shape[4] = {1, 1, 4, 4};
    float image_data[16] = { 
        1,2,3,4,
        2,3,4,1,
        3,4,2,1,
        4,1,2,3
    };
    float expected_stride1[9] = { 
        2,3,3,
        3,3.25,2,
        3,2.25,2
    };

    tensor_fp32 *a = init_with_data(4, image_shape, image_data);
    tensor_fp32* out = op_fp32avgpool2d(a, 2,2, 1, 0);

    bool passed = true;

    if (!check_equals(out->data, expected_stride1, 9)){
        passed = false;
    }


    free_tensor(a);
    free_tensor(out);

    return passed;
}

/*
 * Conv2d tests
 */
bool conv2d_3x3mean_kernel() {
    int image_shape[4] = {1, 1, 5, 5};
    float image_data[25] = { 
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    };

    tensor_fp32 *image = init_with_data(4, image_shape, image_data);

    // create image 3x3 mean kernel
    int kernel_shape[3] = {1, 3,3};
    tensor_fp32 *k = init_with_zeros(3, kernel_shape);
    scalarop_inplace_fp32add(k, 1);
    scalarop_inplace_fp32mul(k, (float) 1/9);

    tensor_fp32* out = op_fp32conv2d(image, k, 1, 0);
    // tensor_fp32* out = op_fp32maxpool2d(image, 2,2, 2, 1);

    float expected[9] = {
        3,4,5,
        4,5,6,
        5,6,7
    };

    bool passed = check_equals(out->data, expected, 9);

    free_tensor(image);
    free_tensor(k);
    free_tensor(out);
    return passed;
}

bool conv2d_2x2mean_kernel() {
    int image_shape[4] = {1, 1, 5, 5};
    float image_data[25] = { 
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9
    };

    tensor_fp32 *image = init_with_data(4, image_shape, image_data);

    // create image 2x2 mean kernel
    int kernel_shape[3] = {1, 2,2};
    tensor_fp32 *k = init_with_zeros(3, kernel_shape);
    scalarop_inplace_fp32add(k, 1);
    scalarop_inplace_fp32mul(k, (float) 1/4);

    tensor_fp32* out = op_fp32conv2d(image, k, 1, 0);

    float expected[16] = {
        2,3,4,5,
        3,4,5,6,
        4,5,6,7,
        5,6,7,8
    };

    bool passed = check_equals(out->data, expected, 16);

    free_tensor(image);
    free_tensor(k);
    free_tensor(out);
    return passed;
}


bool check_equals(float* a, float* b, int size){
    for(int i=0; i<size; i++){
        if(a[i] != b[i]){
            return false;
        }
    }
    return true;
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
