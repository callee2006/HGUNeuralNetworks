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
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "HGULayer.h"

HGULayer::HGULayer(){
	m_inputDim = 0;
	m_outputDim = 0;

	m_pInput = NULL;
	m_aOutput = NULL;
	m_aWeight = NULL;
	m_weightDim = 0;
	m_bWeightShared = FALSE;

	m_aGradient = NULL;
	m_aDelta = NULL;
	m_aDeltaBar = NULL;
}

HGULayer::HGULayer(int inputDim, int outputDim, HGULayer *pShareSrc){
	m_inputDim = 0;
	m_outputDim = 0;

	m_pInput = NULL;
	m_aOutput = NULL;
	m_aWeight = NULL;
	m_weightDim = 0;
	m_bWeightShared = FALSE;

	m_aGradient = NULL;
	m_aDelta = NULL;
	m_aDeltaBar = NULL;

	Alloc(inputDim, outputDim, pShareSrc);
}

int HGULayer::Alloc(int inputDim, int outputDim, HGULayer *pShareSrc)
{
	if(IsAllocated())
		Delete();

	m_pInput = NULL;
	m_aOutput = new float[outputDim];

	m_weightDim = (inputDim + 1) * outputDim;
	if(pShareSrc == NULL){
		try{
			m_aWeight = new float[m_weightDim];
		} catch(...){
			printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
			return FALSE;
		}

		float range = 1.F / (float)sqrt(inputDim + 1.F);
		// init weights by random numbers in [-0.1, +0.1]
		for(int w = 0; w < m_weightDim; w++)
			m_aWeight[w] = rand() / (float)RAND_MAX * 2 * range - range;
		m_bWeightShared = FALSE;
	} else {
		m_aWeight = pShareSrc->GetWeight();
		m_bWeightShared = TRUE;
	}

	try {
		m_aGradient = new float[(inputDim + 1) * outputDim];
		m_aDelta = new float[outputDim];
		m_aDeltaBar = new float[outputDim];
	} catch(...){
		if(m_bWeightShared == FALSE){
			delete[] m_aWeight;
			m_aWeight = NULL;
		}
		printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return FALSE;
	}
	
	m_inputDim = inputDim;
	m_outputDim = outputDim;
	memset(m_aGradient, 0, (inputDim + 1) * outputDim * sizeof(float));		// reset gradients

	return TRUE;
}

void HGULayer::Delete()
{
	if(m_aOutput){
		delete[] m_aOutput;
		m_aOutput = NULL;
	}

	if(m_aWeight){
		if(m_bWeightShared == FALSE)
			delete[] m_aWeight;
		else
			m_bWeightShared = FALSE;
		m_aWeight = NULL;
	}

	if(m_aGradient){ 
		delete[] m_aGradient;
		m_aGradient = NULL;
	}
	if(m_aDelta){
		delete[] m_aDelta;
		m_aDelta = NULL;
	}

	if(m_aDeltaBar){
		delete[] m_aDeltaBar;
		m_aDeltaBar = NULL;
	}

	m_inputDim = 0;
	m_outputDim = 0;
}

void PropagateOnDevice(float *pInput, float *pWeight, int inDim, int outDim, float *pOutput);		// CUDA Code

int HGULayer::Propagate(float *pInput)
{
	m_pInput = pInput;

#ifdef __CUDA__
	return PropagateOnDevice(pInput, m_aWeight, m_inputDim, m_outputDim, m_aOutput);
#endif // __CUDA__
	
	for(int o = 0; o < m_outputDim; o++){
		float net = 0.F;
		float *inWeight = m_aWeight + o * (m_inputDim + 1);
		for(int i = 0; i < m_inputDim; i++)
			net += pInput[i] * inWeight[i];
		net += inWeight[m_inputDim];					// bias

		m_aOutput[o] = Activation(net);
	}

	return TRUE;
}

float HGULayer::GetError(float *pDesiredOutput)
{
	if(pDesiredOutput == NULL)
		return 0.F;

	float error = 0.F;
	for(int i = 0; i < m_outputDim; i++)
		error += (m_aOutput[i] - pDesiredOutput[i]) * (m_aOutput[i] - pDesiredOutput[i]);

	if(m_outputDim > 1)
		error /= m_outputDim;

	return error;
}

int HGULayer::GetMaxOutputIndex()
{
	int maxIdx = 0;
	for(int o = 1; o < m_outputDim; o++){
		if(m_aOutput[o] > m_aOutput[maxIdx])
			maxIdx = o;
	}

	return maxIdx;
}

int HGULayer::ComputeGradient(float *pInput, float *pDesiredOutput)
{
	Propagate(pInput);
	ComputeDeltaBar(pDesiredOutput);
	ComputeGradientFromDeltaBar();

	return TRUE;
}

int HGULayer::ComputeDeltaBar(float *pDesiredOutput)
{
	for(int o = 0; o < m_outputDim; o++)
		m_aDeltaBar[o] = (m_aOutput[o] - pDesiredOutput[o]) / m_outputDim;

	return TRUE;
}

int HGULayer::ComputeGradientFromDeltaBar()
{
	int i = 0, o = 0;

	// compute delta from delta_bar
	for(o = 0; o < m_outputDim; o++)
		m_aDelta[o] = m_aDeltaBar[o] * DerActivationFromOutput(m_aOutput[o]);

	// compute gradient from delta and input
	for(o = 0; o < m_outputDim; o++){
		for(i = 0; i < m_inputDim; i++)
			m_aGradient[(m_inputDim + 1) * o + i] += m_aDelta[o] * m_pInput[i];
		m_aGradient[(m_inputDim + 1) * o + m_inputDim] += m_aDelta[o];		// gradient of bias
	}

	return TRUE;
}

int HGULayer::Backpropagate(float *pPrevDeltaBar)
{
	for(int i = 0; i < m_inputDim; i++){
		pPrevDeltaBar[i] = 0.F;
		for(int o = 0; o < m_outputDim; o++)
			pPrevDeltaBar[i] += m_aWeight[(m_inputDim + 1) * o + i] * m_aDelta[o];
	}

	return TRUE;
}

int HGULayer::UpdateWeight(float learningRate)
{
	for(int o = 0; o < m_outputDim; o++){
		for(int i = 0; i < m_inputDim + 1; i++){
			m_aWeight[o * (m_inputDim + 1) + i] -= learningRate * m_aGradient[o * (m_inputDim + 1) + i];
			m_aGradient[o * (m_inputDim + 1) + i] = 0.F;		// reset gradient
		}
	}

	return TRUE;
}

int HGULayer::UpdateBias(float learningRate)
{
	for(int o = 0; o < m_outputDim; o++){
		m_aWeight[o * (m_inputDim + 1) + m_inputDim] -= learningRate * m_aGradient[o * (m_inputDim + 1) + m_inputDim];
		m_aGradient[o * (m_inputDim + 1) + m_inputDim] = 0.F;
	}

	return TRUE;
}

void HGULayer::ResetGradient()
{
	memset(m_aGradient, 0, m_weightDim * sizeof(m_aGradient[0]));
}

void HGULayer::MergeGradient(HGULayer *pSrc)
{
	float *gSrc = pSrc->GetGradient();
	float *gDest = m_aGradient;
	float *gLimit = m_aGradient + m_weightDim;

	for(; gDest < gLimit; gDest++, gSrc++){
		*gDest += *gSrc;
		*gSrc = 0.F;
	}
//	pSrc->ResetGradient();
}
