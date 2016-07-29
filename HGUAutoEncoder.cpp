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
#include "HGUAutoEncoder.h"


int HGUAutoEncoder::Alloc(int inputDim, int outputDim)
{
	int ret = HGULayer::Alloc(inputDim, outputDim);

	m_pDecoder = new HGULayer(outputDim, inputDim);
	if(ret == FALSE || m_pDecoder == FALSE){
		printf("Failed to allocate layer in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
		return FALSE;
	}
		
	CopyWeightTranspose(m_pDecoder->GetWeight(), outputDim, inputDim, GetWeight());

	return ret;
}

void HGUAutoEncoder::Delete(){
	if(m_pDecoder){
		delete m_pDecoder;
		m_pDecoder = NULL;
	}
	HGULayer::Delete();
}

int HGUAutoEncoder::ComputeGradient(float *pInput)
{
	Reproduce(pInput);
	
	// gradient of decoder
	m_pDecoder->ComputeDeltaBar(pInput);
	m_pDecoder->ComputeGradientFromDeltaBar();

	// gradient of encoder
	m_pDecoder->Backpropagate(GetDeltaBar());
	ComputeGradientFromDeltaBar();

	return TRUE;
}

int HGUAutoEncoder::UpdateWeight(float learningRate)
{
	MergeGradientTranspose(GetGradient(), GetInputDim(), GetOutputDim(), m_pDecoder->GetGradient());
	HGULayer::UpdateWeight(learningRate);
	CopyWeightTranspose(m_pDecoder->GetWeight(), GetOutputDim(), GetInputDim(), GetWeight());
	m_pDecoder->UpdateBias(learningRate);
	return TRUE;
}

void HGUAutoEncoder::MergeGradientTranspose(float *pDest, int inputDim, int outputDim, float *pTrSrc)
{
	for(int i = 0; i < outputDim; i++){
		for(int j = 0; j < inputDim; j++){
			pDest[i * (inputDim + 1) + j] += pTrSrc[j * (outputDim + 1) + i];
			pTrSrc[j * (outputDim + 1) + i] = 0.F;
		}
	}

}

void HGUAutoEncoder::CopyWeightTranspose(float *pDest, int inputDim, int outputDim, float *pTrSrc)
{
	for(int i = 0; i < outputDim; i++){
		for(int j = 0; j < inputDim; j++)
			pDest[i * (inputDim + 1) + j] = pTrSrc[j * (outputDim + 1) + i];
	}
}
