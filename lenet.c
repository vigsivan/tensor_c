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

typedef struct {
    int lbl;
    tensor_fp32 *data;
} mnist_image;

lenet* load_lenet(const char* checkpoint){
    int nmodules;
    int ndims[10];
    int EXPECTED_NDIMS[] = {4, 1, 4, 1, 2, 1, 2, 1, 2, 1};
    tensor_fp32* tensors[10];

    FILE *file = fopen(checkpoint, "rb");
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE); 
    }
    fread(&nmodules, sizeof(int), 1, file);
    if (nmodules != 10){
        fprintf(stderr, "Number of modules does not match (got %i, expected 10).\n", nmodules);
        exit(EXIT_FAILURE); 

    }

    for (int i = 0; i < 10; i++){
        fread(ndims + i, sizeof(int), 1, file);
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

    printf("Successfully loaded checkpoint `%s' with expected number of modules\n", checkpoint);
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

mnist_image* load_mnist(char* mnist_path){
    FILE* file = fopen(mnist_path, "rb");     
    int dims[4] = { 1, 1, 28, 28 };
    tensor_fp32* data = init_tensor(4, dims);
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", mnist_path);
        exit(EXIT_FAILURE); 
    }
    int label;
    fread(&label, sizeof(int), 1, file);
    printf("Read mnist file with label %i\n", label);
    fread(data->data, sizeof(float)*28*28, 1, file);
    mnist_image* image = malloc(sizeof(mnist_image));
    image->lbl = label;
    image->data = data;
    return image;
}


int lenet_forward(lenet *net, tensor_fp32 *input){
    // FIXME: conv2d needs to be re-implemented to allow for a bias term
    tensor_fp32* x = op_fp32conv2d(input, net->c1w, 1, 2);
    return 0;
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
    lenet* net = load_lenet(checkpoint_path);
    mnist_image* mi = load_mnist(mnist_folder);
    int pred = lenet_forward(net, mi->data);
    if (pred != mi->lbl){
        printf("Incorrect prediction: Predicted %i, but label is %i", pred, mi->lbl);
    }
    // print_linear(net->c0w);
    // print_linear(net->c1w);
}
