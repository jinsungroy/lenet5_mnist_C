#ifndef READ_MNIST_H_
#define READ_MNIST_H_

#include <iostream>
using namespace std;

void read_mnist_images(float *mnistData, string full_path);
void read_mnist_labels(int *mnistLabel, string full_path);
#endif
