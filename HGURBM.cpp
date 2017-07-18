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
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "HGULayer.h"
#include "HGURBM.h"


int HGURBM::Alloc(int inputDim, int outputDim, HGULayer *pShareSrc)
{
	if(m_aWeight)
		Delete();

	int weightDim = (inputDim + 1) * (outputDim + 1) - 1;
	try {
		if(pShareSrc == NULL){
			m_aWeight = new float[weightDim];
			m_bWeightShared = FALSE;
		} else {
			m_aWeight = pShareSrc->GetWeight();
			m_bWeightShared = TRUE;
		}
		m_aGradient = new float[weightDim];

		m_aVisible = new float[inputDim];
		m_aVisibleProb = new float[inputDim];
		m_aVisibleClamped = new unsigned char[inputDim];

		m_aOutput = new float[outputDim];
		m_aOutputProb = new float[outputDim];
		m_aOutputProb0 = new float[outputDim];
		m_aOutputClamped = new unsigned char[outputDim];
	} catch(...){
		printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return FALSE;
	}

	// initialize weight by random numbers between -1/sqrt(in_degree) and 1/sqrt(in_degree)
	float range = sqrt(1.F / ((inputDim + outputDim) / 2));

	float *w = m_aWeight;
	for(int o = 0; o < outputDim; o++){
		for(int i = 0; i < inputDim; i++, w++)
			*w = rand() / (float)RAND_MAX * 2 * range - range;
		*(w++) = 0.F;								// bias for output
	}
	for(int i = 0; i < inputDim; i++, w++)
		*w = 0.F;									// bias for input

	memset(m_aGradient, 0, weightDim * sizeof(m_aGradient[0]));
	memset(m_aVisibleClamped, 0, inputDim * sizeof(m_aVisibleClamped[0]));
	memset(m_aOutputClamped, 0, outputDim * sizeof(m_aOutputClamped[0]));

	m_inputDim = inputDim;
	m_outputDim = outputDim;
	m_weightDim = weightDim;

	return TRUE;
}

void HGURBM::Delete()
{
	if(m_aVisible){
		delete[] m_aVisible;
		m_aVisible = NULL;
		delete[] m_aVisibleProb;
		m_aVisibleProb = NULL;
		delete[] m_aVisibleClamped;
		m_aVisibleClamped = NULL;
	}

	if(m_aOutputProb){
		delete[] m_aOutputProb;
		m_aOutputProb = NULL;

		delete[] m_aOutputProb0;
		m_aOutputProb0 = NULL;

		delete[] m_aOutputClamped;
		m_aOutputClamped = NULL;
	}

	if(m_aGroupIndex){
		delete[] m_aGroupIndex;
		m_aGroupIndex = NULL;
		m_noGroup = 0;
	}

	if(IsAllocated())
		HGULayer::Delete();
}

int HGURBM::ComputeOutputProb(float *pInput)
{
	if(!IsAllocated()){
		printf("RBM was not allocated in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return FALSE;
	}

	if(pInput != NULL){
		m_pInput = pInput;
		memcpy(m_aVisible, pInput, m_inputDim * sizeof(m_aVisible[0]));
	}

	float *w = m_aWeight;
	for(int o = 0; o < m_outputDim; o++){
		float net = 0.F;
		float *iLimit = m_aVisible + m_inputDim;
		for(float *ip = m_aVisible; ip < iLimit; ip++, w++)
			net += *w * *ip;
		net += *(w++);				// bias

		m_aOutputProb[o] = Activation(net);
	}

	return TRUE;
}

int HGURBM::ComputeVisibleProb()
{
	if(!IsAllocated()){
		printf("RBM was not allocated in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return FALSE;
	}

	int pitch = m_inputDim + 1;

	for(int i = 0; i < m_inputDim; i++){
		float *w = m_aWeight + i;
		float net = 0.F;
		float *oLimit = m_aOutput + m_outputDim;
		for(float *op = m_aOutput; op < oLimit; op++, w += pitch)
			net += *w * *op;
		net += *w;					// bias
		m_aVisibleProb[i] = Activation(net);
	}

	return TRUE;
}


int HGURBM::GibsSampling(float *pInput, int k, float *pOutputProb0)
{
	if(pInput != NULL){
		m_pInput = pInput;
		memcpy(m_aVisible, m_pInput, m_inputDim * sizeof(m_aVisible[0]));
	}

	for(int i = 0; i < k ; i++){
		Encode(NULL);

		if(i == 0 && pOutputProb0)
			memcpy(m_aOutputProb0, m_aOutputProb, m_outputDim * sizeof(m_aOutputProb0[0]));

		Decode();
	}

	return TRUE;
}

int HGURBM::Sample_Binomial(float *pProb, int size, float *pDigit, unsigned char *pClamped)
{
	float *limit = pDigit + size;

	if(pClamped != NULL){
		for(; pDigit < limit; pProb++, pDigit++, pClamped++){
			if(*pClamped == FALSE){
				float r = rand() / (float)RAND_MAX;
				*pDigit = (r <= *pProb) ? 1.F : 0.F;
			}
		}
	} else {
		for(; pDigit < limit; pProb++, pDigit++){
			float r = rand() / (float)RAND_MAX;
			*pDigit = (r <= *pProb) ? 1.F : 0.F;
		}
	}

	return TRUE;
}

int HGURBM::ComputeGradient_CD(float *pInput, int k)
{
	GibsSampling(pInput, k, m_aOutputProb0);
	ComputeOutputProb(NULL);

	float *g = m_aGradient;
	for(int o = 0; o < m_outputDim; o++){
		for(int i = 0; i < m_inputDim; i++, g++)
			*g += -m_pInput[i] * m_aOutputProb0[o] + m_aVisible[i] * m_aOutputProb[o];
		*(g++) += -m_aOutputProb0[o] + m_aOutputProb[o];	// bias of hidden (output) nodes
	}

	// bias of visible nodes
	for(int i = 0; i < m_inputDim; i++, g++)
		*g += -m_pInput[i] + m_aVisible[i];

	return TRUE;
}

int HGURBM::UpdateWeight(float learningRate)
{
	float *w = m_aWeight;
	float *wLimit = m_aWeight + (m_inputDim + 1) * (m_outputDim + 1) - 1;
	float *g = m_aGradient;

	for(; w < wLimit; w++, g++){
		*w -= learningRate * *g;
		*g = 0.F;
	}

	return TRUE;
}

float HGURBM::GetEnergy(float *pInput)
{
	if(pInput != NULL){
		m_pInput = pInput;
		memcpy(m_aVisible, pInput, m_inputDim * sizeof(m_aVisible[0]));
		ComputeOutputProb(NULL);
	}

	float energy = 0.F;

	float *w = m_aWeight;
	for(int o = 0; o < m_outputDim; o++){
		float op = m_aOutputProb[o];
		for(int i = 0; i < m_inputDim; i++, w++)
			energy -= m_aVisible[i] * *w * op;
		energy -= *w * op;
	}

	for(int i = 0; i < m_inputDim; i++, w++)
		energy -= *w * m_aVisibleProb[i];

	return energy;
}

float HGURBM::GetError(float *pInput)
{
	if(pInput != NULL){
		m_pInput = pInput;
		memcpy(m_aVisible, pInput, m_inputDim * sizeof(m_aVisible[0]));
		ComputeOutputProb(NULL);
	}

	float error = 0.F;
	for(int i = 0; i < m_inputDim; i++)
		error += (m_pInput[i] - m_aVisibleProb[i]) * (m_pInput[i] - m_aVisibleProb[i]);
	if(m_inputDim >= 1)
		error /= 2 * m_inputDim;

	return error;
}

void HGURBM::Display()
{
/*
	printf("Input: ");
	for(int i = 0; i < m_inputDim; i++)
		printf("%1f(%.2f) ", m_pInput[i], m_pInput[i]);
	printf("\n");
*/
	
	printf("Output(hidden): ");
	for(int i = 0; i < m_outputDim; i++)
		printf("%.0f(%.2f) ", m_aOutput[i], m_aOutputProb[i]);
	printf("\n");
	
	printf("Visible(reproduction): ");
	for(int i = 0; i < m_inputDim; i++)
		printf("%.0f(%.2f) ", m_aVisible[i], m_aVisibleProb[i]);
	printf("\n");

}

void HGURBM::DisplayWeight(float *pWeight, const char *title)
{
	if(pWeight == NULL)
		pWeight = m_aWeight;

	printf("%s %d x %d\n", title, m_inputDim, m_outputDim); 

	float *w = pWeight;
	for(int o = 0; o < m_outputDim; o++){
		for(int i = 0; i < m_inputDim + 1; i++, w++)
			printf("%6.3f ", *w);
		printf("\n");
	}

	for(int i = 0; i < m_inputDim; i++, w++)
		printf("%6.3f ", *w);
	printf("\n");
}


