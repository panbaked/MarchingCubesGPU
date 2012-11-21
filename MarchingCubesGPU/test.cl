#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// volume data
__constant sampler_t volumeSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t tableSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void
displayTriBuffer(__read_only image2d_t triTex)
{
	size_t x = get_global_id(0);
	printf("\n %i :",x);
	for(int i = 0; i < 16; i++)
	{
		int v = read_imageui(triTex, tableSampler, (int2)(i, x)).x;
		printf(" %i", v);
	}
}

__kernel void 
computeDensityField(__global float4 * out, float blockSize, uint points)
{
	float4 controlPoint = (float4)(points*blockSize*0.5f, points*blockSize*0.5f, points*blockSize*0.5f, 0);
	
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t z = get_global_id(2);

	int4 coord = (int4)(x, y, z, 0);

	float4 val = (float4)(x*blockSize, y*blockSize, z*blockSize, 0);
	float dist = length(val - controlPoint);
	float isoval = 4*pow(2, -(dist*dist)/100);

	val.w = isoval;

	int index = x + y * points + z * points * points;
	out[index] = val;
	
	//write_imagef(out, coord, val);
}

void
interpolate(float isolevel, float4 a, float4 b, float4 gradientA, float4 gradientB, float4 * p, float4 * n)
{
	float alpha = (isolevel - a.w) / (b.w - a.w);
	*p = mix(a, b, alpha);
	(*n).x = mix(gradientA.x, gradientB.x, alpha);
	(*n).y = mix(gradientA.y, gradientB.y, alpha);
	(*n).z = mix(gradientA.z, gradientB.y, alpha);
}

__kernel void
calculateVertices(__global float4 * volume, float isovalue, __global uint * outVertCount, __global uint * outCubeCount,
					__global int4 * cubes, __global uint2 *occupiedCubes,
					__read_only image2d_t numVertsTex, __local uint * scratch)
{

	//we divide the volume cube into cubes of size 2x2x2 vertices
	//so we counter this by multiplying by two in our index to get the actual index into the volume
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t z = get_global_id(2);


	size_t xLocal = get_local_id(0);
	size_t yLocal = get_local_id(1);
	size_t zLocal = get_local_id(2);
	size_t localSize = get_local_size(0);

	size_t size = get_global_size(0);
	size_t vConvSize = (size_t)pow((float)size, 2.0f); //conversion factor to index into volume from cubes

	uint cubeVertIndices[8];
	cubeVertIndices[0] = x + y * vConvSize + z * vConvSize * vConvSize;
	cubeVertIndices[1] = x + y * vConvSize + (z + 1) * vConvSize * vConvSize;
	cubeVertIndices[2] = (x + 1) + y * vConvSize + (z + 1) * vConvSize * vConvSize;
	cubeVertIndices[3] = (x + 1) + y * vConvSize + z * vConvSize * vConvSize;
	cubeVertIndices[4] = x + (y + 1) * vConvSize + z * vConvSize * vConvSize;
	cubeVertIndices[5] = x + (y + 1) * vConvSize + (z + 1) * vConvSize * vConvSize;
	cubeVertIndices[6] = (x + 1) + (y + 1) * vConvSize + (z + 1) * vConvSize * vConvSize;
	cubeVertIndices[7] = (x + 1) + (y + 1) * vConvSize + z * vConvSize * vConvSize;
		 /*
	for(int i = 0; i < 8; i++)
	{
		printf("%i, ", cubeVertIndices[i]);
	}
	printf("\n");
	*/
	float4 cubeVertices[8];
	cubeVertices[0] = volume[cubeVertIndices[0]];
	cubeVertices[1] = volume[cubeVertIndices[1]];
	cubeVertices[2] = volume[cubeVertIndices[2]];
	cubeVertices[3] = volume[cubeVertIndices[3]];
	cubeVertices[4] = volume[cubeVertIndices[4]];
	cubeVertices[5] = volume[cubeVertIndices[5]];
	cubeVertices[6] = volume[cubeVertIndices[6]];
	cubeVertices[7] = volume[cubeVertIndices[7]];

	/*
	cubeVertices[0] = read_imagef(volume, volumeSampler, coord*2);
	cubeVertices[1] = read_imagef(volume, volumeSampler, coord*2 + (int4)(0, 0, 1, 0));
	cubeVertices[2] = read_imagef(volume, volumeSampler, coord*2 + (int4)(1, 0, 1, 0));
	cubeVertices[3] = read_imagef(volume, volumeSampler, coord*2 + (int4)(1, 0, 0, 0));
	cubeVertices[4] = read_imagef(volume, volumeSampler, coord*2 + (int4)(0, 1, 0, 0));
	cubeVertices[5] = read_imagef(volume, volumeSampler, coord*2 + (int4)(0, 1, 1, 0));
	cubeVertices[6] = read_imagef(volume, volumeSampler, coord*2 + (int4)(1, 1, 1, 0));
	cubeVertices[7] = read_imagef(volume, volumeSampler, coord*2 + (int4)(1, 1, 0, 0));
	*/
	//printf("coord %i, %i, %i ! %f, %f, %f, %f, %f, %f, %f, %f \n ", coord.x, coord.y, coord.z
//		,cubeVertices[0].w, cubeVertices[1].w, cubeVertices[2].w, cubeVertices[3].w,
	//	cubeVertices[4].w, cubeVertices[5].w, cubeVertices[6].w, cubeVertices[7].w);

	int cubeIndex = 0;
	cubeIndex |= (cubeVertices[0].w < isovalue);
	cubeIndex |= (cubeVertices[1].w < isovalue)*2;
	cubeIndex |= (cubeVertices[2].w < isovalue)*4;
	cubeIndex |= (cubeVertices[3].w < isovalue)*8;
	cubeIndex |= (cubeVertices[4].w < isovalue)*16;
	cubeIndex |= (cubeVertices[5].w < isovalue)*32;
	cubeIndex |= (cubeVertices[6].w < isovalue)*64;
	cubeIndex |= (cubeVertices[7].w < isovalue)*128;


	uint vertCount = read_imageui(numVertsTex, tableSampler, (int2)(cubeIndex, 0)).x;

	uint reductionIndex = x + y * size/localSize + z * size/localSize * size/localSize; 
	uint localIndex = xLocal+yLocal*localSize+zLocal*localSize*localSize;
	uint cubePosition = x+y*size + z*size*size;

	uint populated = (vertCount > 0);

	occupiedCubes[cubePosition].x = populated;
	occupiedCubes[cubePosition].y = vertCount;

	cubes[cubePosition] = (int4)(x, y, z, cubeIndex);

	scratch[localIndex] = vertCount;

	barrier(CLK_LOCAL_MEM_FENCE);

	if(localIndex == 0)
	{
		uint result = 0;
		for(int i = 0; i < localSize*localSize*localSize; i++)
		{
			result += scratch[i];
		}
		
		outVertCount[reductionIndex] = result;
		if(result > 0)
		{
		//	printf("%i, %i, %i ! %i, %i, %i;", x, y, z, index, cubePosition, result);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
}
__kernel void 
computePrefixArray(__global uint2 *occupiedCubes, __global uint2 *prefixArray, uint size)
{
	uint count = 0;
	uint vertCount = 0;
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t z = get_global_id(2);

	uint i = x + y * size + z * size * size;
	for(uint j = 0; j < i; j++)
	{
		if(occupiedCubes[j].x > 0)
		{
		
			vertCount += occupiedCubes[j].y;
			count++;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
//	printf("%i %i, %i \n", i, vertCount, count);
	prefixArray[i].x = count;
	prefixArray[i].y = vertCount;
}
__kernel void
compactCubes(__global int4 * compactedCubeArray, __global uint2 * prefixArray, __global int4 * cubes, __global uint * vertPrefixArray, uint size)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t z = get_global_id(2);

	uint j = x + y * size + z * size * size;

	uint i = prefixArray[j].x;

	if(cubes[j].w > 0 && cubes[j].w < 255)
	{
		compactedCubeArray[i] = cubes[j];
		vertPrefixArray[i] = prefixArray[j].y;
	}
}
__kernel void
generateTriangles(__global float4 * volume, float isovalue, __global float4 * outVerts, __global float4 * outNormals,
					__global int4 * compactedCubeArray, __global uint * vertPrefixArray, __read_only image2d_t numVertsTex, 
					__read_only image2d_t triTex, uint size)
{
	size_t index = get_global_id(0);

	size_t vConvSize = (size_t)pow((float)size, 2.0f); //volume conversion factor to index into volume from cubes

	uint cubeVertIndices[8];
	cubeVertIndices[0] = compactedCubeArray[index].x + compactedCubeArray[index].y * vConvSize + compactedCubeArray[index].z * vConvSize * vConvSize;
	cubeVertIndices[1] = compactedCubeArray[index].x + compactedCubeArray[index].y * vConvSize + (compactedCubeArray[index].z + 1) * vConvSize * vConvSize;
	cubeVertIndices[2] = (compactedCubeArray[index].x + 1) + compactedCubeArray[index].y * vConvSize + (compactedCubeArray[index].z + 1) * vConvSize * vConvSize;
	cubeVertIndices[3] = (compactedCubeArray[index].x + 1) + compactedCubeArray[index].y * vConvSize + compactedCubeArray[index].z * vConvSize * vConvSize;
	cubeVertIndices[4] = compactedCubeArray[index].x + (compactedCubeArray[index].y + 1) * vConvSize + compactedCubeArray[index].z * vConvSize * vConvSize;
	cubeVertIndices[5] = compactedCubeArray[index].x + (compactedCubeArray[index].y + 1) * vConvSize + (compactedCubeArray[index].z + 1) * vConvSize * vConvSize;
	cubeVertIndices[6] = (compactedCubeArray[index].x + 1) + (compactedCubeArray[index].y + 1) * vConvSize + (compactedCubeArray[index].z + 1) * vConvSize * vConvSize;
	cubeVertIndices[7] = (compactedCubeArray[index].x + 1) + (compactedCubeArray[index].y + 1) * vConvSize + compactedCubeArray[index].z * vConvSize * vConvSize;
		 

	float4 cubeVertices[8];
	cubeVertices[0] = volume[cubeVertIndices[0]];
	cubeVertices[1] = volume[cubeVertIndices[1]];
	cubeVertices[2] = volume[cubeVertIndices[2]];
	cubeVertices[3] = volume[cubeVertIndices[3]];
	cubeVertices[4] = volume[cubeVertIndices[4]];
	cubeVertices[5] = volume[cubeVertIndices[5]];
	cubeVertices[6] = volume[cubeVertIndices[6]];
	cubeVertices[7] = volume[cubeVertIndices[7]];


	int cubeIndex = 0;
	cubeIndex |= (cubeVertices[0].w < isovalue);
	cubeIndex |= (cubeVertices[1].w < isovalue)*2;
	cubeIndex |= (cubeVertices[2].w < isovalue)*4;
	cubeIndex |= (cubeVertices[3].w < isovalue)*8;
	cubeIndex |= (cubeVertices[4].w < isovalue)*16;
	cubeIndex |= (cubeVertices[5].w < isovalue)*32;
	cubeIndex |= (cubeVertices[6].w < isovalue)*64;
	cubeIndex |= (cubeVertices[7].w < isovalue)*128;

	float4 vertList[12];
	float4 normalList[12];

	float4 gradient0 = (float4)(0, 0, 0, 0);
	float4 gradient1 = (float4)(1, 1, 1, 1);

	interpolate(isovalue, cubeVertices[0], cubeVertices[1], gradient0, gradient1, &vertList[0], &normalList[0]);
	interpolate(isovalue, cubeVertices[1], cubeVertices[2], gradient0, gradient1, &vertList[1], &normalList[1]);
	interpolate(isovalue, cubeVertices[2], cubeVertices[3], gradient0, gradient1, &vertList[2], &normalList[2]);
	interpolate(isovalue, cubeVertices[3], cubeVertices[0], gradient0, gradient1, &vertList[3], &normalList[3]);
	interpolate(isovalue, cubeVertices[4], cubeVertices[5], gradient0, gradient1, &vertList[4], &normalList[4]);
	interpolate(isovalue, cubeVertices[5], cubeVertices[6], gradient0, gradient1, &vertList[5], &normalList[5]);
	interpolate(isovalue, cubeVertices[6], cubeVertices[7], gradient0, gradient1, &vertList[6], &normalList[6]);
	interpolate(isovalue, cubeVertices[7], cubeVertices[4], gradient0, gradient1, &vertList[7], &normalList[7]);
	interpolate(isovalue, cubeVertices[0], cubeVertices[4], gradient0, gradient1, &vertList[8], &normalList[8]);
	interpolate(isovalue, cubeVertices[1], cubeVertices[5], gradient0, gradient1, &vertList[9], &normalList[9]);
	interpolate(isovalue, cubeVertices[2], cubeVertices[6], gradient0, gradient1, &vertList[10], &normalList[10]);
	interpolate(isovalue, cubeVertices[3], cubeVertices[7], gradient0, gradient1, &vertList[11], &normalList[11]);
	
	barrier(CLK_LOCAL_MEM_FENCE);

	uint numVerts = read_imageui(numVertsTex, tableSampler, (int2)(cubeIndex, 0)).x;
	uint vertIndex = vertPrefixArray[index] + 1;

	for(int i = 0; i < numVerts; i +=3)
	{
		outVerts[vertIndex+i] = vertList[read_imageui(triTex, tableSampler, (int2)(i, cubeIndex)).x];
		outNormals[vertIndex+i] = normalList[read_imageui(triTex, tableSampler, (int2)(i, cubeIndex)).x];
		outVerts[vertIndex+i+1] = vertList[read_imageui(triTex, tableSampler, (int2)(i+1, cubeIndex)).x];
		outNormals[vertIndex+i+1] = normalList[read_imageui(triTex, tableSampler, (int2)(i+1, cubeIndex)).x];
		outVerts[vertIndex+i+2] = vertList[read_imageui(triTex, tableSampler, (int2)(i+2, cubeIndex)).x];
		outNormals[vertIndex+i+2] = normalList[read_imageui(triTex, tableSampler, (int2)(i+2, cubeIndex)).x];

	}

}

