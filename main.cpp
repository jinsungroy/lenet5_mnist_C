/* Copyright: Jinsung Kim */
/* Incheon National University */
/* APC & SoC LAB */
/* Email: jinsungroy@inu.ac.kr */


#include <iostream>
#include <fstream>
#include <numeric>
#include <time.h>
#include <vector>
#include "read_mnist.h"
#include "layers.h"
#include <stdint.h>
#include "classify_lib.h"
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
using namespace std;

void load_model(string filename, float* weight, int size)
{
	ifstream file(filename.c_str(), ios::in);
	if (file.is_open())
	{
		for (int i = 0; i < size; i++)
		{
			float temp = 0.0;
			file >> temp;
			weight[i] = temp;
		}
	}
	else
	{
		//std::cout<<"Loading model is failed : "<<filename<<endl;
	}
}

int main (void)
{
	clock_t start_point, end_point, c1_start,c1_stop, c2_start,c2_stop,c3_start,c3_stop;
	vector<clock_t> v_c1,v_c2,v_c3;

	// Read MNIST IMAGES, LABELS
	float *mnistData = (float *)malloc(1024*10000*sizeof(float));
	int *mnistLabel = (int *)malloc(10000*sizeof(int));

	read_mnist_images(mnistData, "./dataset/t10k-images.idx3-ubyte");
	read_mnist_labels(mnistLabel, "./dataset/t10k-labels.idx1-ubyte");

		
	// Read Layer parameters
	float* Wconv1 = (float *)malloc(5*5*1*6*sizeof(float));
	float* Wconv2 = (float *)malloc(5*5*6*16*sizeof(float));
	float* Wconv3 = (float *)malloc(5*5*16*120*sizeof(float));
	float* Wfc2 = (float *)malloc(120 * 84 * sizeof(float));
	float* Wfc3 = (float *)malloc(84 * 10 * sizeof(float));
	float* bconv1 = (float *)malloc(6*sizeof(float));
	float* bconv2 = (float *)malloc(16*sizeof(float));
	float* bconv3 = (float *)malloc(120*sizeof(float));
	float* bfc2 = (float *)malloc(84*sizeof(float));
	float* bfc3 = (float *)malloc(10*sizeof(float));

	if(!Wconv1||!Wconv2||!Wconv3||!bconv1||!bconv2||!bconv3||!Wfc2||!Wfc3||!bfc2||!bfc3)
	{
		cout<<"mem alloc error(1)"<<endl;
		exit(1);
	}

	cout<<"Load models"<<endl;
	load_model("./parameters/Wconv1.mdl",Wconv1,5*5*6);
	load_model("./parameters/Wconv2.mdl",Wconv2,5*5*6*16);
	load_model("./parameters/Wconv3.mdl",Wconv3,5*5*16*120);
	load_model("./parameters/Wfc2.mdl",Wfc2,120*84);
	load_model("./parameters/Wfc3.mdl",Wfc3,84*10);
	load_model("./parameters/bconv1.mdl",bconv1,6);
	load_model("./parameters/bconv2.mdl",bconv2,16);
	load_model("./parameters/bconv3.mdl",bconv3,120);
	load_model("./parameters/bfc2.mdl",bfc2,84);
	load_model("bfc3.mdl",bfc3,10);
	cout<<"model loaded"<<endl;

	// Memory allocation
	float* inputImage		= (float*) malloc(32*32*sizeof(float));
	float* outputConv1 		= (float*) malloc(6*28*28*sizeof(float));
	float* outputPool1 		= (float*) malloc(6*14*14*sizeof(float));
	float* outputConv2 		= (float*) malloc(16*10*10*sizeof(float));
	float* outputPool2 		= (float*) malloc(16*5*5*sizeof(float));
	float* outputPool2_flatten = (float*)malloc(16 * 5 * 5 * sizeof(float));
	float* outputConv3 		= (float*) malloc(120*sizeof(float));
	float* outputFc2 		= (float*) malloc(84*sizeof(float));
	float* outputFc3 		= (float*) malloc(10*sizeof(float));
	if(!inputImage || !outputConv1 || !outputPool1 || !outputConv2 || !outputPool2 || !outputConv3 || !outputFc2 || !outputFc3){
		cout<<"Memory allocation error(2)"<<endl;
		exit(1);
	}


	vector<double> result_hw;
	double accuracy_hw;
	int init=1;
	for(int i=0; i<10000; i++)
	{
		for(int batch=0;batch<1024;batch++)
		{
			inputImage[batch] = mnistData[i*1024 + batch];
		}

		convolution(inputImage, Wconv1, bconv1, outputConv1, 6, 1, 5, 28, 32);
		maxPooling(outputConv1, outputPool1, 6, 28);

		convolution(outputPool1,Wconv2,bconv2,outputConv2, 16, 6, 5, 10, 14);
		maxPooling(outputConv2, outputPool2, 16, 10);
		
		flatten(outputPool2, outputPool2_flatten, 16, 25);
		fullyConnected(outputPool2_flatten,Wconv3,bconv3,outputConv3, 120, 400);

		fullyConnected(outputConv3,Wfc2,bfc2,outputFc2, 84, 120);

		fullyConnected(outputFc2,Wfc3,bfc3,outputFc3, 10, 84);

		result_hw.push_back(equal(mnistLabel[i], argmax(outputFc3)));

		for (int i=0; i<10; i++)
		{
			printf("value [%d]: %f\n", i, outputFc3[i]);
		}

		printf("mnistLabel[%d]: %d\n", i, mnistLabel[i]);
		printf("argmax: %d\n", argmax(outputFc3));
		printf("Result: %d\n\n", equal(mnistLabel[i],argmax(outputFc3)));
	}
	accuracy_hw = 1.0*accumulate(result_hw.begin(),result_hw.end(),0.0);
	cout<<"Inference completed"<<endl;
	cout<<"accuracy : "<<accuracy_hw<<"/"<<result_hw.size()<<endl;


	free(inputImage);
	free(outputConv1);
	free(outputPool1);
	free(outputConv2);
	free(outputPool2);
	free(outputPool2_flatten);
	free(outputConv3);
	free(outputFc2);
	free(outputFc3);

	free(Wconv1);
	free(Wconv2);
	free(Wconv3);
	free(bconv1);
	free(bconv2);
	free(bconv3);
	free(Wfc2);
	free(bfc2);
	free(Wfc3);
	free(bfc3);

	free(mnistData);
	free(mnistLabel);

	cout<<"Test Completed"<<endl;

	return 0;
}
