#pragma OPENCL EXTENSION cl_khr_fp64: enable

#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

//calculate_hidden_layer
__kernel void kernel1(const __global double *input, const __global double *weights, __global double *hidden_layer, const int input_size, const int hidden_size) {
    int i = get_global_id(0);
    
    double sum = 0;
    for (int j = 0; j < input_size; j++) {
        sum += input[j] * weights[j * hidden_size + i];
    }
    hidden_layer[i] = sum;
}

//calculate_output_layer
__kernel void kernel2(const __global double *hidden_layer, const __global double *weights, __global double *output_layer, const int hidden_size, const int output_size) {
    int i = get_global_id(0);
    
    double sum = 0;
    for (int j = 0; j < hidden_size; j++) {
        sum += hidden_layer[j] * weights[j * output_size + i];
    }
    output_layer[i] = sum;
}

//calculate_error_output
__kernel void kernel3(const __global double *output_layer, const double expected, __global double *error_output, const int output_size) {
    int i = get_global_id(0);
    //for (int j = 0; j < output_size; j++) 
     error_output[i] = output_layer[i] - expected;
}

//calculate_error_hidden
__kernel void kernel4(const __global double *error_output, const __global double *hidden_layer, const __global double *weights, __global double *error_hidden, const int hidden_size, const int output_size) {
    int i = get_global_id(0);
    
    double sum = 0;
    for (int j = 0; j < output_size; j++) {
        sum += error_output[j] * weights[i * output_size + j];
    }
    error_hidden[i] = sum * hidden_layer[i] * (1 - hidden_layer[i]);
}

//update_weights
__kernel void kernel5(__global double* error_output, __global double* error_hidden, __global double* hidden_layer, __global double* input, __global double* weights_hidden, __global double* weights_output, const int hidden_size, const int output_size) {
	int i = get_global_id(0);
	double learning_rate = 0.1;

	for (int j = 0; j < output_size; j++) {
    		weights_output[i * output_size + j] -= learning_rate * error_output[j] * hidden_layer[i];
		}

	for (int j = 0; j < hidden_size; j++) {
    		weights_hidden[i * hidden_size + j] -= learning_rate * error_hidden[j] * input[i];
		}
}