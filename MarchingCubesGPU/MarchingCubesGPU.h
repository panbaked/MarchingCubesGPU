#pragma once
#ifndef MARCHINGCUBESGPU_H_
#define MARCHINGCUBESGPU_H_


#include "CL/cl.hpp"
#include "CL/opencl.h"
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cstdio>
#include <fstream>
#include "MarchingCubesTables.h"

const size_t VOLUME_SIZE = 32;
const size_t MARCH_SIZE = VOLUME_SIZE/2;
const size_t TOTAL_CUBES = MARCH_SIZE*MARCH_SIZE*MARCH_SIZE;

typedef cl_float4 (*Cube)[VOLUME_SIZE][VOLUME_SIZE][VOLUME_SIZE];

class MCGPU
{
public:
	MCGPU(void);
	~MCGPU(void);
	int update(void);
private:
	size_t marchSize;
	size_t volumeSize;
	cl::Image2D triTableImage;
	cl::Image2D vertTableImage;
	cl::Buffer volumeBuffer;
	std::vector<cl_float4> volume;
	cl::Buffer cubes;
	cl::Buffer occupiedCubes;
	cl::Buffer prefixBuffer;

	cl::Context context;
	cl::Program program;
	cl::CommandQueue queue;

	int setupCL(void);
	int allocateBuffers();
	int generateDensityField();
	int march();
	int dispose(void);
};

#endif