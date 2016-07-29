/***
	Copyright 2012 Injung Kim

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
***/


#include <stdio.h>

#include <cuda_runtime.h>

__global__ void Propagate_kernel(float *pDevInput, float *pDevWeight, int inDim, int outDim, float *pDevOutput);

void PropagateOnDevice(float *pInput, float *pWeight, int inDim, int outDim, float *pOutput)
{
	float *pDevInput = NULL, *pDevOutput = NULL, *pDevWeight = NULL;

	// allocate device memory	
	cudaMalloc((void**)&pDevInput, inDim * sizeof(float));
	cudaMalloc((void**)&pDevOutput, outDim * sizeof(float));
	cudaMalloc((void**)&pDevWeight, (inDim+1)*outDim * sizeof(float));
	
	// copy input and weight to device memory
	cudaMemcpy(pDevInput, pInput, inDim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pDevWeight, pWeight, (inDim+1)*outDim * sizeof(float), cudaMemcpyHostToDevice);
	
	// compute output on CUDA device
	Propagate_kernel<<<1, outDim>>>(pDevInput, pDevWeight, inDim, outDim, pDevOutput);
	
	// copy output to host memory
	cudaMemcpy(pOutput, pDevOutput, outDim * sizeof(float), cudaMemcpyDeviceToHost);
	
	// deallocate device memory
	cudaFree(pDevInput);
	cudaFree(pDevOutput);
	cudaFree(pDevWeight);
}

__global__ void Propagate_kernel(float *pDevInput, float *pDevWeight, int inDim, int outDim, float *pDevOutput)
{
	// idx is thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < outDim){
		float *w = &pDevWeight[idx * (inDim + 1)];

		float net = 0.F;
		for(int i = 0; i < inDim; i++)
				net += pDevInput[i] * w[i];
		net += w[inDim];						// add bias

		pDevOutput[idx] = 1.F/(1.F + (float)exp(-net));

		idx += blockDim.x * gridDim.x; 	//	blockDim.x * gridDim.x is # of threads
	}
}
