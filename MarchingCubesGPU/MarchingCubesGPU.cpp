// MarchingCubesGPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "MarchingCubesGPU.h"

inline void checkErr(cl_int err, const char* name)
{
	if(err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ") " << std::endl;
		int i = 0;
		std::cin >> i; 
		exit(EXIT_FAILURE);
	}

}

const std::string hw("Hello World");
const size_t width = 32;
const size_t height = 32;
const size_t depth = 32;

int MCGPU::update(void)
{
	int err = generateDensityField();
	err = march();
	return err;
}

MCGPU::MCGPU(void)
{
	this->setupCL();
}

MCGPU::~MCGPU(void)
{
	//blah
}
int MCGPU::setupCL(void)
{
	cl_int err;
    std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
    checkErr(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
    std::cerr << "Platform number is: " << platformList.size() << std::endl;
    
    std::string platformVendor;
    platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
    std::cerr << "Platform is by: " << platformVendor << "\n";
    cl_context_properties cprops[3] = 
        {CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
 
	this->context = cl::Context(
       CL_DEVICE_TYPE_GPU, 
       cprops,
       NULL,
       NULL,
       &err);
    checkErr(err, "Conext::Context()");   

	//Device setup and program build
	std::vector<cl::Device> devices;
	devices = this->context.getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	cl::STRING_CLASS version = devices[0].getInfo<CL_DEVICE_VERSION>();

	std::cout << version << std::endl;

	std::ifstream file("test.cl");
	checkErr(file.is_open() ? CL_SUCCESS : -1, "file load error");

	std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));

	this->program = cl::Program(this->context, source);
	err = program.build(devices);
	
	if(err == CL_BUILD_PROGRAM_FAILURE)
	{
		cl::STRING_CLASS str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
		std::cout << " \n\t\t\tBUILD LOG\n";
        std::cout << " ************************************************\n";
        std::cout << str.c_str() << std::endl;
        std::cout << " ************************************************\n";
	}
	
	cl_device_type deviceName = devices[0].getInfo<CL_DEVICE_TYPE>();

	std::cout << deviceName << std::endl;

	this->queue = cl::CommandQueue(this->context, devices[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");

	err = this->allocateBuffers();
	checkErr(err, "Buffer allocationg failed.");

	return err;
}
int MCGPU::allocateBuffers()
{
	int err;
	//Lookup tables
	this->triTableImage = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), 256, 16, 0, (void*) triTable, &err);
	checkErr(err, "Tritable image failed");

	this->vertTableImage = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), 256, 1, 0, (void*) numVertsTable, &err);
	checkErr(err, "Verttable image failed");

	//Volume map
	this->volume = std::vector<cl_float4>(VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE);

	this->volumeBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(cl_float4), 0, &err);
	checkErr(err, "Volume read image failed");

	//Cube buffers
	this->occupiedCubes = cl::Buffer(context, CL_MEM_READ_WRITE, TOTAL_CUBES*sizeof(cl_uint2), 0, &err);
	checkErr(err, "Occupied cubes failed");

	this->prefixBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, TOTAL_CUBES*sizeof(cl_uint2), 0, &err);
	checkErr(err, "Prefix buffer failed");

	this->cubes = cl::Buffer(context, CL_MEM_READ_WRITE, TOTAL_CUBES*sizeof(cl_int4), 0, &err);
	checkErr(err, "Cubes buffer failed");

	return err;
}

int MCGPU::generateDensityField()
{
	//density field calculations
	cl_int err;

	cl::Kernel computeDensityKernel(this->program, "computeDensityField", &err);
	checkErr(err, "computeDensityKernel failed to initialize");

	err = computeDensityKernel.setArg(0, this->volumeBuffer);
	checkErr(err, "computeDensityKernel Kernel::setArg(0)");

	err = computeDensityKernel.setArg(1, 1.0f);
	checkErr(err, "computeDensityKernel Kernel::setArg(1)");

	err = computeDensityKernel.setArg(2, VOLUME_SIZE);
	checkErr(err, "computeDensityKernel Kernel::setArg(2)");

	err = this->queue.enqueueNDRangeKernel(computeDensityKernel, cl::NullRange, cl::NDRange(width, height, depth), cl::NDRange(1, 1, 1));
	checkErr(err, "computeDensityKernel CommandQueue::enqueueNDRangeKernel()");

	err = this->queue.finish();
	checkErr(err, "computeDensityKernel finish failed");

	this->queue.flush();

	err = this->queue.enqueueReadBuffer(this->volumeBuffer, CL_TRUE, 0, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(cl_float4), &volume[0], NULL, NULL);
	checkErr(err, "density image read fail");

	queue.flush();
	checkErr(err, "Density flush failed.");

	std::cout << "Generated the density field." << std::endl;

	return err;
	
}

int MCGPU::march()
{
	int err;	
	const size_t WORK_GROUP_DIM = 4;
	const size_t CUBE_REDUCTION = MARCH_SIZE/WORK_GROUP_DIM;
	const size_t WORK_GROUPS = CUBE_REDUCTION*CUBE_REDUCTION*CUBE_REDUCTION;

	cl::Kernel computeVertsKernel(this->program, "calculateVertices", &err);
	checkErr(err, "computeVertsKernel failed to initialize");

	err = computeVertsKernel.setArg(0, volumeBuffer);
	checkErr(err, "computeVertsKernel setArg(0)");
	
	err = computeVertsKernel.setArg(1, 0.5f);
	checkErr(err, "computeVertsKernel setArg(1)");

	std::vector<cl_uint> vertCountArray = std::vector<cl_uint>(WORK_GROUPS);
	
	cl::Buffer outVertCountBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, WORK_GROUPS*sizeof(cl_uint), 0, &err);
	checkErr(err, "Out vert count buffer failed.");

	cl::Buffer outCubeCountBuffer = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, WORK_GROUPS*sizeof(cl_uint));
	checkErr(err, "Out cube count buffer failed.");

	err = computeVertsKernel.setArg(2, outVertCountBuffer);
	checkErr(err, "computeVertsKernel setArg(2)");

	err = computeVertsKernel.setArg(3, outCubeCountBuffer);
	checkErr(err, "computeVertsKernel setArg(3)");

	err = computeVertsKernel.setArg(4, this->cubes);
	checkErr(err, "computeVertsKernel setArg(4)");

	err = computeVertsKernel.setArg(5, this->occupiedCubes);
	checkErr(err, "computeVertsKernel setArg(5)");

	err = computeVertsKernel.setArg(6, vertTableImage);
	checkErr(err, "computeVertsKernel setArg(6)");

	err = computeVertsKernel.setArg(7, 4*4*4*sizeof(cl_uint), NULL);
	checkErr(err, "computeVertsKernel setArg(7)");

	err = this->queue.enqueueNDRangeKernel(computeVertsKernel, cl::NullRange, cl::NDRange(MARCH_SIZE, MARCH_SIZE, MARCH_SIZE), cl::NDRange(WORK_GROUP_DIM, WORK_GROUP_DIM, WORK_GROUP_DIM));
	checkErr(err, "computeVertsKernel enqueue failed");

	err = this->queue.finish();
	checkErr(err, "computeVertsKernel queue wait failed");

	err = this->queue.flush();
	checkErr(err, "computeVertsKernel queue flush failed");
	
	err = this->queue.enqueueReadBuffer(outVertCountBuffer, CL_TRUE, 0,  WORK_GROUPS*sizeof(cl_uint), &vertCountArray[0]);
	checkErr(err, "out vert count could not be fetched");

	std::vector<cl_uint2> occpiedCubesPtr = std::vector<cl_uint2>(TOTAL_CUBES);

	err = this->queue.enqueueReadBuffer(occupiedCubes, CL_TRUE, 0, TOTAL_CUBES*sizeof(cl_uint2), &occpiedCubesPtr[0]);
	checkErr(err, "occupied cubes could not be fetched");

	
	int vertCount = 0;
	for(int i = 0; i < WORK_GROUPS; i++)
	{
		vertCount += vertCountArray[i];
	}

	std::cout << "Generated " << vertCount << " vertices." << std::endl;

	queue.flush();

	cl::Kernel prefixKernel(program, "computePrefixArray", &err);

	err = prefixKernel.setArg(0, occupiedCubes);
	err = prefixKernel.setArg(1, prefixBuffer);
	err = prefixKernel.setArg(2, MARCH_SIZE);

	err = this->queue.enqueueNDRangeKernel(prefixKernel, cl::NullRange, cl::NDRange(MARCH_SIZE, MARCH_SIZE, MARCH_SIZE), cl::NDRange(1, 1, 1));
	err = this->queue.finish();
	err = this->queue.flush();

	std::vector<cl_uint2> prefixArray = std::vector<cl_uint2>(TOTAL_CUBES);
	err = queue.enqueueReadBuffer(prefixBuffer, CL_TRUE, 0, TOTAL_CUBES*sizeof(cl_uint2), &prefixArray[0]);

	cl_uint cubeCount = prefixArray[prefixArray.size()-1].s[0];

	std::cout << "Found " << cubeCount << " cubes to march." << std::endl;

	cl::Kernel compactKernel(this->program, "compactCubes", &err);

	cl::Buffer compactedCubesBuffer(context, CL_MEM_READ_WRITE, cubeCount * sizeof(cl_int4), 0, &err);
	cl::Buffer compactedVertPrefixBuffer(context, CL_MEM_READ_WRITE, cubeCount * sizeof(cl_uint), 0, &err);

	err = compactKernel.setArg(0, compactedCubesBuffer);
	err = compactKernel.setArg(1, prefixBuffer);
	err = compactKernel.setArg(2, cubes);
	err = compactKernel.setArg(3, compactedVertPrefixBuffer);
	err = compactKernel.setArg(4, MARCH_SIZE);

	err = queue.enqueueNDRangeKernel(compactKernel, cl::NullRange, cl::NDRange(MARCH_SIZE, MARCH_SIZE, MARCH_SIZE), cl::NDRange(1, 1, 1));
	err = queue.finish();

	std::vector<cl_int8> compactedCubes = std::vector<cl_int8>(cubeCount);

	err = queue.enqueueReadBuffer(compactedCubesBuffer, CL_TRUE, 0, cubeCount * sizeof(cl_int4), &compactedCubes[0]);

	std::cout << "Generated the compacted cube array." << std::endl;

	cl::Kernel marchKernel(this->program, "generateTriangles", &err);


	cl::Buffer vertexBuffer(this->context, CL_MEM_READ_WRITE, vertCount * sizeof(cl_float4), 0, &err);
	cl::Buffer normalBuffer(this->context, CL_MEM_READ_WRITE, vertCount * sizeof(cl_float4), 0, &err);

	err = marchKernel.setArg(0, volumeBuffer);
	err = marchKernel.setArg(1, 0.5f);
	err = marchKernel.setArg(2, vertexBuffer);
	err = marchKernel.setArg(3, normalBuffer);
	err = marchKernel.setArg(4, compactedCubesBuffer);
	err = marchKernel.setArg(5, compactedVertPrefixBuffer);
	err = marchKernel.setArg(6, vertTableImage);
	err = marchKernel.setArg(7, triTableImage);
	err = marchKernel.setArg(8, MARCH_SIZE);

	err = queue.enqueueNDRangeKernel(marchKernel, cl::NullRange, cl::NDRange(cubeCount), cl::NDRange(1));

	std::vector<cl_float4> vertices = std::vector<cl_float4>(vertCount);
	err = queue.enqueueReadBuffer(vertexBuffer, CL_TRUE, 0, vertCount * sizeof(cl_float4), &vertices[0]);

	std::cout << "Done marching." << std::endl;

	return err;

}
int _tmain(int argc, _TCHAR* argv[])
{

	MCGPU mcGPU = MCGPU();
	int err = mcGPU.update();
	
	return EXIT_SUCCESS;
}
