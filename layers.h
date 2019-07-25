/* Copyright: Jinsung Kim */
/* Incheon National University */
/* APC & SoC LAB */
/* Email: jinsungroy@inu.ac.kr */

#ifndef IMAGE_POOL_H_
#define IMAGE_POOL_H_

void convolution(const float *input_feature, const float *weights, const float *bias, float *output_feature, int output_channel, int input_channel,
	int k_size, int output_wh, int input_wh);
void fullyConnected(const float *input_feature, const float *weights, const float *bias, float *output_feature, int output_channel, int input_channel);
void flatten(const float *input_feature, float *output_feature, int input_channel, int input_wh);
void maxPooling(const float *input_feature, float *output_feature, int input_channel, int input_wh);
float relu(float x);
float _tanh(float x);
#endif