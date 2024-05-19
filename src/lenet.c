#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <dirent.h>
#include <string.h>

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

void lenet_free(lenet* net){
    free_tensor(net->c0w);
    free_tensor(net->c0b);
    free_tensor(net->c1w);
    free_tensor(net->c1b);
    free_tensor(net->l0w);
    free_tensor(net->l0b);
    free_tensor(net->l1w);
    free_tensor(net->l1b);
    free_tensor(net->l2w);
    free_tensor(net->l2b);
    free(net);
}

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
        tensors[i] = init_tensor(ndims_i, dims, NULL);
    }

    for (int i = 0; i < 10; i++){
        tensor_fp32* t = tensors[i];
        fread(t->data, sizeof(float) * t->size, 1, file);
    }

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
    tensor_fp32* data = T(1,1,28,28);
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", mnist_path);
        exit(EXIT_FAILURE); 
    }
    int label;
    fread(&label, sizeof(int), 1, file);
    fread(data->data, sizeof(float)*28*28, 1, file);
    mnist_image* image = malloc(sizeof(mnist_image));
    image->lbl = label;
    image->data = data;
    return image;
}

tensor_fp32* sse(int num_classes, int lbl, tensor_fp32* prediction){
    if (prediction->ndims != 2 || prediction->dims[0] != 1 || prediction->dims[1] != num_classes){
        fprintf(stderr, "Expected shape (1, %d). Got %dD tensor with last dim shape %d\n", num_classes, prediction->ndims, prediction->dims[prediction->ndims-1]);
        exit(1);
    }
    tensor_fp32* lbl_tensor = T(1,num_classes);
    setindex(lbl_tensor, 1, 0, lbl);
    tensor_fp32* out = scalarop_fp32exp(op_fp32sub(prediction, lbl_tensor), 2);
    out = op_fp32total(out);
    return out;
}

tensor_fp32* lenet_forward(lenet *net, tensor_fp32* input){
    tensor_fp32* x = op_fp32conv2d(input, net->c0w, net->c0b, 1, 2);
    x = op_fp32sigmoid(x);
    x = op_fp32avgpool2d(x, 2, 2, 2, 0);
    x = op_fp32conv2d(x, net->c1w, net->c1b, 1, 0);
    x = op_fp32sigmoid(x);
    x = op_fp32avgpool2d(x, 2, 2, 2, 0);
    x = op_fp32flatten(x);

    x = op_fp32linear(x, net->l0w, net->l0b);
    x = op_fp32sigmoid(x);
    x = op_fp32linear(x, net->l1w, net->l1b);
    x = op_fp32sigmoid(x);
    x = op_fp32linear(x, net->l2w, net->l2b);
    x = op_fp32sigmoid(x);

    return x;
}

int lenet_inference(lenet *net, tensor_fp32 *input){
	tensor_fp32* x = lenet_forward(net, input);

    // TODO: implement argmax for tensor_fp32
    float max = -INFINITY;
    int argmax = -1;
    for(int i =0; i < x->size; i++){
        if (x->data[i] > max){
            max = x->data[i];
            argmax = i;
        }
    }

    return argmax;
}


void error_usage() {
    fprintf(stderr, "Usage:   lenet <checkpoint> <mnist_folder>\n");
    fprintf(stderr, "Example: lenet lenet.bin /home/jovyan/mnist_folder");
    exit(EXIT_FAILURE);
}


int main(int argc, char** argv) {
    char* checkpoint_path = NULL;
    char* mnist_folder = NULL;
    struct dirent *de;  // Pointer for directory entry 

    if (argc < 3) {
        error_usage();
    }

    checkpoint_path = argv[1];
    mnist_folder = argv[2];
    lenet* net = load_lenet(checkpoint_path);

  
    DIR *dr = opendir(mnist_folder); 
    if (dr == NULL)  // opendir returns NULL if couldn't open directory 
    { 
        printf("Could not open mnist directory %s.\n", mnist_folder); 
        return 0; 
    } 
    int correct = 0;
    int total = 0;
    int count = 15;
    while ((de = readdir(dr)) != NULL) {
        count -=1;
        if (de->d_name[0] == '.'){
            continue;
        }

        char path[1024];
        int len = snprintf(path, sizeof(path)-1, "%s/%s", mnist_folder, de->d_name);
        path[len] = 0;
        mnist_image* mi = load_mnist(path);
        tensor_fp32* pred = lenet_forward(net, mi->data);
        tensor_fp32* loss = sse(10, mi->lbl, pred);
        backward(loss);
    }
    closedir(dr);     

    printf("Got %d correct predictions out of %d", correct ,total);
    lenet_free(net);
}
