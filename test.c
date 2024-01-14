#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "tensor.h"

void print_data(tensor_fp32 *t);

/*
 * Test functions
 */
bool maxpool2d_test();

// TODO: Add more tests
int main(){
    srand(42);
    printf("Running tests...\n");
    if(!maxpool2d_test()){
        printf("maxpool2d test failed\n");
        return 1;
    }
}

bool maxpool2d_test(){
    int image_shape[4] = {1, 1, 5, 5};
    tensor_fp32 *a = init_with_random(4, image_shape);
    tensor_fp32* out = op_fp32maxpool2d(a, 2,2, 2, 1);
    // free(a); free(out);
    print_2d(a);
    print_2d(out);
    free_tensor(a);
    free_tensor(out);
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
