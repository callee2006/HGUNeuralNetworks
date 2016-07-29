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
#include <math.h>

#include "HGULayer.h"
#include "HGUNeuralNetwork.h"

int FindMaxIndex(float *pArray, int size);

int HGUNeuralNetwork::Alloc(int noLayer, int *pNoNode)
{
	if(IsAllocated())
		Delete();

	m_aLayer = new HGULayer[noLayer];
	if(m_aLayer == NULL){
		printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return FALSE;
	}
	for(int i = 0; i < noLayer; i++){
		int ret = m_aLayer[i].Alloc(pNoNode[i], pNoNode[i+1]);
		if(ret == FALSE){
			printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
			delete[] m_aLayer;
			m_aLayer = NULL;
			return FALSE;
		}
	}

	m_noLayer = noLayer;

	return TRUE;
}

void HGUNeuralNetwork::Delete()
{
	if(m_aLayer){
		delete[] m_aLayer;
		m_aLayer = NULL;
	}
	m_noLayer = 0;
}

int HGUNeuralNetwork::Propagate(float *pInput)
{
	if(IsAllocated() == FALSE){
		printf("HGUNeuralNetwork was not allocated!\n");			
		return FALSE;
	}

	m_aLayer[0].Propagate(pInput);
	for(int i = 1; i < m_noLayer; i++)
		m_aLayer[i].Propagate(m_aLayer[i-1].GetOutput());
	
	return TRUE;
}

int HGUNeuralNetwork::ComputeGradient(float *pInput, float *pDesiredOutput)
{
	if(IsAllocated() == FALSE){
		printf("HGUNeuralNetwork was not allocated!\n");			
		return FALSE;
	}

	Propagate(pInput);

	for(int i = m_noLayer - 1; i >= 0; i--){
		if(i == m_noLayer-1)
			m_aLayer[i].ComputeDeltaBar(pDesiredOutput);
		else
			m_aLayer[i+1].Backpropagate(m_aLayer[i].GetDeltaBar());

		m_aLayer[i].ComputeGradientFromDeltaBar();
	}

	return TRUE;
}

int HGUNeuralNetwork::UpdateWeight(float learningRate)
{
	if(IsAllocated() == FALSE){
		printf("HGUNeuralNetwork was not allocated!\n");			
		return FALSE;
	}

	for(int i = 0; i < m_noLayer; i++)
		m_aLayer[i].UpdateWeight(learningRate);

	return TRUE;
}

float HGUNeuralNetwork::GetError(float *pDesiredOutput)
{
	if(IsAllocated() == FALSE){
		printf("HGUNeuralNetwork was not allocated!\n");			
		return 0.F;
	}

	return m_aLayer[m_noLayer-1].GetError(pDesiredOutput);
}