// include libraries
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

//Global variables 
const int INPUT_SIZE = 3;
const int HIDDEN_SIZE = 10;
const int OUTPUT_SIZE = 2;
const int DATASET_SIZE = 5;
const double LEARNING_RATE = 0.1;

//functions 

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivative_sigmoid(double x) {
    return x * (1.0 - x);
}

std::vector<double> calculate_hidden_layer(const std::vector<double>& input, const std::vector<std::vector<double>>& weights) {
    std::vector<double> hidden_layer(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * weights[j][i];
        }
        hidden_layer[i] = sigmoid(sum);
    }
    return hidden_layer;
}

std::vector<double> calculate_output_layer(const std::vector<double>& hidden_layer,const std::vector<std::vector<double>>& weights) {
    std::vector<double> output_layer(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden_layer[j] * weights[j][i];
        }
        output_layer[i] = sigmoid(sum);
    }
    return output_layer;
}

std::vector<double> calculate_error_output(const std::vector<double>& output_layer,double expected) {
    std::vector<double> error_output(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        error_output[i] = (expected - output_layer[i]) * derivative_sigmoid(output_layer[i]);
    }
    return error_output;
}

std::vector<double> calculate_error_hidden(const std::vector<double>& error_output,const std::vector<double>& hidden_layer,const std::vector<std::vector<double>>& weights) {
    std::vector<double> error_hidden(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += error_output[j] * weights[i][j];
        }
        error_hidden[i]= derivative_sigmoid(hidden_layer[i]) * sum;
    }
    return error_hidden;
}

void update_weights(std::vector<std::vector<double>>& weights_input_to_hidden, std::vector<std::vector<double>>& weights_hidden_to_output, const std::vector<double>& input, const std::vector<double>& hidden_layer, const std::vector<double>& error_output, const std::vector<double>& error_hidden) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_input_to_hidden[j][i] += LEARNING_RATE * error_hidden[i] * input[j];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_hidden_to_output[j][i] += LEARNING_RATE * error_output[i] * hidden_layer[j];
        }
    }
}
std::vector<std::vector<double>> dataset = {
    	  {1, -2, 1},
        {2, -4, 2},
        {3, -6, 3},
        {4, -8, 4},
        {5, -10, 5},
};


//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;

	// Create a context for the device
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	//for cpu code:
	std::vector<std::vector<double>> weights_input_to_hidden(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
    std::vector<std::vector<double>> weights_hidden_to_output(HIDDEN_SIZE, std::vector<double>(OUTPUT_SIZE));
    
    // Initialize weights with random values
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_input_to_hidden[i][j] = rand() / (double)RAND_MAX;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_hidden_to_output[i][j] = rand() / (double)RAND_MAX;
        }
    }
    //Training 
	for (int iteration = 0; iteration < 1000; iteration++) {
        for (int d = 0; d < DATASET_SIZE; d++) {
            // Feedforward
            std::vector<double> input = dataset[d];
            std::vector<double> hidden_layer = calculate_hidden_layer(input, weights_input_to_hidden);
            std::vector<double> output_layer = calculate_output_layer(hidden_layer, weights_hidden_to_output);
            
            // Calculate error
            double expected = -dataset[d][1] / (2 * dataset[d][0]);
            std::vector<double> error_output = calculate_error_output(output_layer, expected);
            std::vector<double> error_hidden = calculate_error_hidden(error_output, hidden_layer, weights_hidden_to_output);
            
            // Update weights
            update_weights(weights_input_to_hidden, weights_hidden_to_output, input, hidden_layer, error_output, error_hidden);
        }
    }
    //Testing
    std::vector<std::vector<double>> test_data = {{1, -2, 1}, {3, -6, 3}, {2, -4, 2}};
    for (int i = 0; i < test_data.size(); i++) {
        std::vector<double> test_input = test_data[i];
        std::vector<double> hidden_layer = calculate_hidden_layer(test_input, weights_input_to_hidden);
        std::vector<double> output_layer = calculate_output_layer(hidden_layer, weights_hidden_to_output);
    
        std::cout << "Estimated result for test case " << i + 1 << ": " << output_layer[0] << std::endl;
    }

	//end of cpu code 

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/nn.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel kernel(program, "kernel");
/*
	// Declare some values
	std::size_t wgSize = 128; // Number of work items per work group
	std::size_t count = wgSize * 100000; // Overall number of work items = Number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for input data and for output data from CPU and GPU on the host
	//std::vector<float> h_input (count);
	//std::vector<float> mem_weights_hidden (count);
	
	
	// Allocate space for input and output data on the device
	cl::Buffer mem_input (context, CL_MEM_READ_ONLY, INPUT_SIZE * sizeof(double));
	cl::Buffer mem_weights_hidden (context, CL_MEM_READ_ONLY, INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
	cl::Buffer mem_weights_output (context,  CL_MEM_READ_ONLY, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
	cl::Buffer mem_hidden_layer (context,  CL_MEM_READ_WRITE, HIDDEN_SIZE * sizeof(double));
	cl::Buffer mem_output_layer (context, CL_MEM_READ_WRITE, OUTPUT_SIZE * sizeof(double));
	cl::Buffer mem_error_output (context, CL_MEM_READ_WRITE, OUTPUT_SIZE * sizeof(double));
	cl::Buffer mem_error_hidden (context, CL_MEM_READ_WRITE, HIDDEN_SIZE * sizeof(double));

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	//memset(mem_input.data(), 255, size);
	//memset(h_outputCpu.data(), 255, size);
	//memset(h_outputGpu.data(), 255, size);
//queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	//queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

	// Initialize input data with more or less random values
	for (std::size_t i = 0; i < count; i++)
		h_input[i] = ((i * 1009) % 31) * 0.1;

	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	//calculateHost(h_input, h_outputCpu);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();

	// Copy input data to device
	//TODO: enqueueWriteBuffer()
	cl::Event copy1;
	//queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &copy1);

	// Launch kernel on the device
	cl::Event execution;
	//kernel.setArg<cl::Buffer>(0, d_input);
	//kernel.setArg<cl::Buffer>(1, d_output);
	queue.enqueueNDRangeKernel(kernel, 0, count, wgSize, NULL, &execution);

	// Copy output data back to host
	//TODO: enqueueReadBuffer()
	cl::Event copy2;
	//queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &copy2);

	// Print performance data
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(execution);
	Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(copy1);
	Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(copy2);
	Core::TimeSpan copyTime = copyTime1 + copyTime2;
	Core::TimeSpan overallGpuTime = gpuTime + copyTime;
	std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;

	// Check whether results are correct
	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < count; i++) {
		// Allow small differences between CPU and GPU results (due to different rounding behavior)
		if (!(std::abs (h_outputCpu[i] - h_outputGpu[i]) <= 10e-5)) {
			if (errorCount < 15)
				std::cout << "Result for " << i << " is incorrect: GPU value is " << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i] << std::endl;
			else if (errorCount == 15)
				std::cout << "..." << std::endl;
			errorCount++;
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << "Success" << std::endl;
*/
	return 0;
}
