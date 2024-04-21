#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    tensor_fp32 *c0w;
    tensor_fp32 *c0b;
    tensor_fp32 *c1w;
    tensor_fp32 *c1b;
    tensor_fp32 *l0w;
    tensor_fp32 *l0b;
    tensor_fp32 *l1w;
    tensor_fp32 *l1b;
    tensor_fp32 *l2w;
    tensor_fp32 *l2b;
} lenet;

lenet* load_lenet(const char* checkpoint){
    int nmodules[1];
    int ndims[10];
    int EXPECTED_NDIMS[] = {4, 1, 4, 1, 2, 1, 2, 1, 2, 1};
    tensor_fp32* tensors[10];

    FILE *file = fopen(checkpoint, "rb");
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE); 
    }
    fread(nmodules, sizeof(int), 1, file);
    if (*nmodules != 10){
        fprintf(stderr, "Number of modules does not match (got %i, expected 10).\n", *nmodules);
        exit(EXIT_FAILURE); 

    }

    for (int i = 0; i < 10; i++){
        fread(ndims + i, sizeof(int), 1, file);
    }

    for (int i = 0; i < 10; i++){
        if (ndims[i] != EXPECTED_NDIMS[i]){
            fprintf(stderr, "Number of dimensions for module %i does not match (got %i, expected %i).\n", i, ndims[i], EXPECTED_NDIMS[i]);
            exit(EXIT_FAILURE); 
        }
    }
    
    // initialize tensors
    for (int i = 0; i < 10; i++){
        int ndims_i = ndims[i];
        int dims[ndims_i];
        for(int j=0; j<ndims_i; j++){
            fread(dims + j, sizeof(int), 1, file);
        }
        tensors[i] = init_tensor(ndims_i, dims);
    }

    for (int i = 0; i < 10; i++){
        tensor_fp32* t = tensors[i];
        fread(t->data, sizeof(float) * t->size, 1, file);
    }

    printf("Successfully loaded file with expected number of modules\n");
    fclose(file);

    lenet* net = malloc(sizeof(lenet));

    net->c0w = tensors[0];
    net->c0b = tensors[1];
    net->c1w = tensors[2];
    net->c1b = tensors[3];
    net->l0w = tensors[4];
    net->l0b = tensors[5];
    net->l1w = tensors[6];
    net->l1b = tensors[7];
    net->l2w = tensors[8];
    net->l2b = tensors[9];

    return net;
}

// void lenet_free(lenet *net){
//     free(net->c1);
//     free(net->c2);
//     free(net->d1);
//     free(net->d2);
//     free(net->d3);
//     free(net);
// }
//


void lenet_forward(lenet *net, tensor_fp32 *input){

}


void error_usage() {
    fprintf(stderr, "Usage:   lenet <checkpoint> <mnist_folder>\n");
    fprintf(stderr, "Example: lenet lenet.bin /home/jovyan/mnist_folder");
    exit(EXIT_FAILURE);
}


void main(int argc, char** argv) {
    char* checkpoint_path = NULL;
    char* mnist_folder = NULL;

    if (argc < 3) {
        error_usage();
    }

    checkpoint_path = argv[1];
    mnist_folder = argv[2];
    load_lenet(checkpoint_path);
}
