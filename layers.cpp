/* Copyright: Jinsung Kim */
/* Incheon National University */
/* APC & SoC LAB */
/* Email: jinsungroy@inu.ac.kr */

#include <stdio.h>
#include <math.h>

float relu(float x)
{
	return x > 0 ? x : 0;
}
float _tanh(float x)
{
	float exp2x = expf(2 * x) + 1;
	return (exp2x - 2) / (exp2x);
}

void convolution(const float *input_feature, const float *weights, const float *bias, float *output_feature, int output_channel, int input_channel,
	int k_size, int output_wh, int input_wh)
{
	// input_channel: 
	// output_channel:
	// k_size:
	// input_wh:
	// output_wh:

	for (int row = 0; row < output_wh; row++) {
		for (int col = 0; col < output_wh; col++) {
			for (int output_filter = 0; output_filter < output_channel; output_filter++) {
				float temp = 0;
				for (int input_filter = 0; input_filter < input_channel; input_filter++) {
					for (int kernel_row = 0; kernel_row < k_size; kernel_row++) {
						for (int kernel_col = 0; kernel_col < k_size; kernel_col++) {
							temp = temp + (input_feature[input_filter * input_wh * input_wh + (row + kernel_row) * input_wh + (col + kernel_col)]
								* weights[output_filter * input_channel * k_size * k_size + input_filter * k_size * k_size +
								kernel_row * k_size + kernel_col]);
						}
					}
				}
				output_feature[output_filter * output_wh * output_wh + row * output_wh + col] = relu(temp + bias[output_filter]);
			}
		}
	}
	return;
}

void fullyConnected(const float *input_feature, const float *weights, const float *bias, float *output_feature, int output_channel, int input_channel)
{
	for (int output_filter = 0; output_filter < output_channel; output_filter++) {
		float temp = 0;
		for (int input_filter = 0; input_filter < input_channel; input_filter++) {
			temp = temp + (input_feature[input_filter] * weights[input_filter * output_channel + output_filter]);
		}
		output_feature[output_filter] = relu(temp + bias[output_filter]);
	}
	return;
}

void flatten(const float *input_feature, float *output_feature, int input_channel, int featuremap_size)
{
	for (int i = 0; i < featuremap_size; i++)
	{
		for (int j = 0; j < input_channel; j++)
		{
			output_feature[i * input_channel + j] = input_feature[j * featuremap_size + i];
		}
	}
}

void maxPooling(const float *input_feature, float *output_feature, int input_channel, int input_wh)
{
	for (int depth = 0; depth < input_channel; depth++)
	{
		for (int row = 0; row < input_wh / 2; row++)
		{
			for (int col = 0; col < input_wh / 2; col++)
			{
				float max1, max2, max;
				float array00, array01, array10, array11;

				// Which one is the max value?
				array00 = input_feature[depth * input_wh*input_wh + (2 * row) * input_wh + (2 * col)];
				array01 = input_feature[depth * input_wh*input_wh + (2 * row) * input_wh + (2 * col) + 1];
				array10 = input_feature[depth * input_wh*input_wh + ((2 * row) + 1) * input_wh + (2 * col)];
				array11 = input_feature[depth * input_wh*input_wh + ((2 * row) + 1) * input_wh + (2 * col) + 1];

				max1 = array00 > array01 ? array00 : array01;
				max2 = array10 > array11 ? array10 : array11;
				max = max1 > max2 ? max1 : max2;

				// Write the max value on the output layer
				output_feature[depth * (input_wh / 2) * (input_wh / 2) + row * (input_wh / 2) + col] = max;
			}
		}
	}
}